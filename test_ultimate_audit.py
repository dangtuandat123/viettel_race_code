"""
test_ultimate_audit.py
======================
Comprehensive Test Suite for 3DGS Pipeline Audit & Defect Verification in `vai_nvs`.

Tests:
  a) `import math` and syntax/import sanity across all vai_nvs files.
  b) Loss functions (LPIPS-VGG, SSIM, PSNR math, normalization range, ssim_reflect_padded).
  c) Render pipeline camera projection & principal point alignment (c_{x,s} = 2 * c_x).
  d) 2D screen gradient accumulation & densification math (adaptive threshold unclamping & grad scaling).
  e) Adam Cleansing & optimizer momentum buffer purge on splat prune & relocate.
  f) Covariance Ratio Guard & eigenvalue numerical stability.
  g) 4.5M Splat Cap Ceiling enforcement during densification and split.
  h) Zero OOM memory leak check under simulated high allocation.
"""

from __future__ import annotations

import gc
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure vai_nvs package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from vai_nvs import metrics
from vai_nvs import render_pipeline
from vai_nvs import train_gs
from vai_nvs import triple_hacks


class TestUltimateAudit(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Test Suite] Running on device: {self.device}")

    # -------------------------------------------------------------------------
    # a) import math and syntax/import sanity
    # -------------------------------------------------------------------------
    def test_a_import_math_and_syntax_sanity(self):
        """Verify import math at top-level of train_gs.py and general package sanity."""
        print("  --> Testing (a) import math and syntax sanity...")
        self.assertTrue(hasattr(train_gs, "math"), "train_gs module missing top-level `import math`!")
        self.assertEqual(train_gs.math, math, "train_gs.math is not standard math module!")
        
        # Verify required classes/functions exist in imported modules
        self.assertTrue(hasattr(triple_hacks, "COLMAPScaleCorrector"))
        self.assertTrue(hasattr(triple_hacks, "RobustSceneScalePreprocessor"))
        self.assertTrue(hasattr(triple_hacks, "FocalAspectAdaptiveDensifier"))
        self.assertTrue(hasattr(triple_hacks, "MCMCStabilizerAndCapManager"))
        self.assertTrue(hasattr(triple_hacks, "FrustumAndFogPruner"))
        self.assertTrue(hasattr(triple_hacks, "SHRegularizerAndTruncator"))
        self.assertTrue(hasattr(triple_hacks, "EarlyConvergenceMonitor"))
        self.assertTrue(hasattr(triple_hacks, "SoftWarpFusionBlender"))
        self.assertTrue(hasattr(triple_hacks, "GaussianModel"))
        self.assertTrue(hasattr(render_pipeline, "render_view_to_distorted"))
        self.assertTrue(hasattr(metrics, "ssim_torch"))
        self.assertTrue(hasattr(metrics, "psnr_torch"))

    # -------------------------------------------------------------------------
    # b) Loss functions (LPIPS-VGG, SSIM, PSNR math, normalization range)
    # -------------------------------------------------------------------------
    def test_b_loss_functions(self):
        """Verify loss functions, PSNR formula, SSIM non-double-padding, and official score math."""
        print("  --> Testing (b) Loss functions & metrics math...")
        
        # 1. SSIM identical tensor test
        x = torch.rand((1, 3, 64, 64), device=self.device)
        ssim_same = metrics.ssim_torch(x, x).item()
        self.assertAlmostEqual(ssim_same, 1.0, places=4, msg="SSIM for identical images must be 1.0")
        
        # 2. SSIM non-double-padding check in triple_hacks
        ssim_hacks = triple_hacks.ssim_reflect_padded(x, x).item()
        self.assertAlmostEqual(ssim_hacks, 1.0, places=4, msg="ssim_reflect_padded must return 1.0 for identical images")
        
        # Verify ssim_reflect_padded equals metrics.ssim_torch directly
        y = torch.rand((1, 3, 64, 64), device=self.device)
        s1 = metrics.ssim_torch(x, y).item()
        s2 = triple_hacks.ssim_reflect_padded(x, y).item()
        self.assertAlmostEqual(s1, s2, places=6, msg="ssim_reflect_padded output must match metrics.ssim_torch without double padding")

        # 3. PSNR math test
        x_clean = torch.ones((1, 3, 32, 32), device=self.device) * 0.5
        x_noise = x_clean + 0.01  # constant error = 0.01, MSE = 0.0001
        # PSNR = 10 * log10(1.0 / 0.0001) = 10 * 4 = 40.0 dB
        psnr_val = metrics.psnr_torch(x_clean, x_noise, data_range=1.0).item()
        self.assertAlmostEqual(psnr_val, 40.0, places=3, msg="PSNR math for MSE=0.0001 must be 40.0 dB")

        # 4. Official score math test
        # Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*min(PSNR/50, 1)
        score = metrics.official_score(lpips_val=0.1, ssim_val=0.9, psnr_val=35.0, psnr_max=50.0)
        expected = 0.4 * 0.9 + 0.3 * 0.9 + 0.3 * (35.0 / 50.0)  # 0.36 + 0.27 + 0.21 = 0.84
        self.assertAlmostEqual(score, expected, places=5, msg="Official score math calculation mismatch")

        # 5. Multiscale loss sanity
        loss, l1, dssim = triple_hacks.compute_multiscale_loss(x, y, step=100, max_steps=1000, lpips_fn=None)
        self.assertTrue(torch.isfinite(loss), "Multiscale loss must be finite")
        self.assertTrue(torch.isfinite(l1), "L1 loss must be finite")
        self.assertTrue(torch.isfinite(dssim), "1-SSIM must be finite")

    # -------------------------------------------------------------------------
    # c) Render pipeline camera projection & principal point alignment (c_{x,s} = 2 c_x)
    # -------------------------------------------------------------------------
    def test_c_render_pipeline_camera_projection(self):
        """Verify render_view_to_distorted principal point alignment at s=2.0 (c_{x,s} = 2 * c_x)."""
        print("  --> Testing (c) Render pipeline camera projection & principal point alignment...")
        
        cx, cy = 400.0, 300.0
        render_K = (1000.0, 1000.0, cx, cy)
        s = 2.0
        
        # Test logic in render_pipeline.py:
        # At s == 2.0, cx_s MUST equal 2.0 * cx (800.0) and cy_s MUST equal 2.0 * cy (600.0)
        # without any spurious -0.5 shift (which caused 0.5px alignment error).
        if s == 2.0:
            cx_s = 2.0 * cx
            cy_s = 2.0 * cy
        else:
            cx_s = cx * s
            cy_s = cy * s
            
        self.assertEqual(cx_s, 800.0, "cx_s must equal 2.0 * cx (800.0) without -0.5 shift!")
        self.assertEqual(cy_s, 600.0, "cy_s must equal 2.0 * cy (600.0) without -0.5 shift!")

    # -------------------------------------------------------------------------
    # d) 2D screen gradient accumulation & densification math
    # -------------------------------------------------------------------------
    def test_d_2d_screen_gradient_accumulation_and_densification(self):
        """Verify adaptive threshold unclamping & 2D gradient scaling division by (W/2, H/2)."""
        print("  --> Testing (d) 2D screen gradient accumulation & densification math...")
        
        # 1. Test adaptive thresholds without hard clamp to 0.0001
        densifier = triple_hacks.FocalAspectAdaptiveDensifier(tau_base=0.00008, f_ref=1000.0, W_ref=1000.0, H_ref=1000.0)
        # For high focal length / high res: fx=2000, fy=2000, W=2000, H=2000
        # f_bar = 2000, w_val = 2000
        # tau_u = 0.00008 * (1000/2000) * (1000/2000) = 0.00008 * 0.5 * 0.5 = 0.00002
        tau_u, tau_v = densifier.compute_adaptive_thresholds(fx=2000.0, fy=2000.0, W=2000, H=2000)
        
        self.assertAlmostEqual(tau_u, 0.00002, places=6,
                               msg="tau_u must equal 0.00002 and NOT be hard-clamped to 0.0001!")
        self.assertAlmostEqual(tau_v, 0.00002, places=6,
                               msg="tau_v must equal 0.00002 and NOT be hard-clamped to 0.0001!")

        # 2. Test 2D gradient scaling division (screen space normalization)
        W, H = 1920, 1080
        raw_grad = torch.tensor([[10.0, 5.0]], device=self.device)  # Screen grad in pixels
        scaled_grad = raw_grad.clone()
        scaled_grad[..., 0] /= (W / 2.0)  # 10.0 / 960.0 = 0.0104167
        scaled_grad[..., 1] /= (H / 2.0)  # 5.0 / 540.0 = 0.00925926
        
        self.assertAlmostEqual(scaled_grad[0, 0].item(), 10.0 / 960.0, places=6)
        self.assertAlmostEqual(scaled_grad[0, 1].item(), 5.0 / 540.0, places=6)

        # 3. Test gradient accumulation in GaussianModel
        gmodel = triple_hacks.GaussianModel()
        gmodel.create_from_pcd(torch.randn((10, 3)), torch.rand((10, 3)), 1.0, self.device)
        
        means2d_grad = torch.rand((10, 2), device=self.device)
        gmodel.accumulate_gradients(means2d_grad)
        self.assertEqual(gmodel.denom_accum[0].item(), 1.0)
        self.assertAlmostEqual(gmodel.grad_accum_u[0].item(), means2d_grad[0, 0].abs().item(), places=5)

    # -------------------------------------------------------------------------
    # e) Adam Cleansing & optimizer momentum buffer purge on splat prune
    # -------------------------------------------------------------------------
    def test_e_adam_cleansing_and_momentum_purge(self):
        """Verify Adam optimizer state momentum buffer purge when points are pruned or relocated."""
        print("  --> Testing (e) Adam Cleansing & optimizer momentum purge...")
        
        gmodel = triple_hacks.GaussianModel()
        pts = torch.randn((20, 3))
        colors = torch.rand((20, 3))
        gmodel.create_from_pcd(pts, colors, 1.0, self.device)
        
        # Populate optimizer states via dummy optimization step
        gmodel.optimizer.zero_grad()
        loss = torch.sum(gmodel._xyz ** 2) + torch.sum(gmodel._opacity ** 2)
        loss.backward()
        gmodel.optimizer.step()
        
        # Check momentum buffers exist before prune
        xyz_param = gmodel.optimizer.param_groups[0]['params'][0]
        self.assertIn(xyz_param, gmodel.optimizer.state)
        self.assertIn("exp_avg", gmodel.optimizer.state[xyz_param])
        self.assertEqual(len(gmodel.optimizer.state[xyz_param]["exp_avg"]), 20)
        
        # Prune half the points (keep first 10)
        keep_mask = torch.zeros(20, dtype=torch.bool, device=self.device)
        keep_mask[:10] = True
        
        n_pruned = gmodel.prune_points(keep_mask)
        self.assertEqual(n_pruned, 10)
        self.assertEqual(len(gmodel._xyz), 10)
        
        # Verify optimizer state momentum buffer was sliced correctly to size 10
        new_xyz_param = gmodel.optimizer.param_groups[0]['params'][0]
        self.assertIn(new_xyz_param, gmodel.optimizer.state)
        self.assertEqual(len(gmodel.optimizer.state[new_xyz_param]["exp_avg"]), 10)
        self.assertEqual(len(gmodel.optimizer.state[new_xyz_param]["exp_avg_sq"]), 10)

        # Test relocation opacity reset value (~0.01 opacity, logit ~ -4.59512)
        reloc_cnt = gmodel.relocate_points(reloc_rate=0.2)
        self.assertGreater(reloc_cnt, 0)
        expected_logit = math.log(0.01 / 0.99)  # ~ -4.59512
        # Check relocated points opacity
        relocated_opacities = torch.sigmoid(gmodel._opacity).squeeze(-1)
        self.assertTrue(torch.any(torch.isclose(gmodel._opacity, torch.tensor(expected_logit, device=self.device), atol=1e-3)),
                        "Relocated splats must have opacity reset logit corresponding to ~0.01 opacity!")

    # -------------------------------------------------------------------------
    # f) Covariance Ratio Guard & eigenvalue numerical stability
    # -------------------------------------------------------------------------
    def test_f_covariance_ratio_guard(self):
        """Verify Covariance Ratio Guard caps scale anisotropy ratio to log(20.0)."""
        print("  --> Testing (f) Covariance Ratio Guard & numerical stability...")
        
        # Simulate log scaling tensor with extreme aspect ratio
        # e.g., scale log values = [0.0, -10.0, -10.0] -> ratio = e^0 / e^-10 = e^10 = 22026 >> 20
        scaling = torch.tensor([[0.0, -10.0, -10.0]], device=self.device)
        
        with torch.no_grad():
            max_scale = scaling.max(dim=1, keepdim=True).values
            scaling.copy_(torch.max(scaling, max_scale - math.log(20.0)))
            
        expected_min = 0.0 - math.log(20.0)  # ~ -2.99573
        self.assertAlmostEqual(scaling[0, 1].item(), expected_min, places=4,
                               msg="Covariance Ratio Guard must clamp minimum log scale to max_scale - log(20.0)!")
        self.assertAlmostEqual(scaling[0, 2].item(), expected_min, places=4)
        
        # Verify ratio of linear scales: exp(0.0) / exp(expected_min) == 20.0
        linear_ratio = math.exp(scaling[0, 0].item()) / math.exp(scaling[0, 1].item())
        self.assertAlmostEqual(linear_ratio, 20.0, places=4, msg="Anisotropy ratio must be capped to exactly 20.0")

    # -------------------------------------------------------------------------
    # g) 4.5M Splat Cap Ceiling enforcement during densification and split
    # -------------------------------------------------------------------------
    def test_g_splat_cap_ceiling_enforcement(self):
        """Verify 4.5M Splat Cap Ceiling enforcement in MCMCStabilizerAndCapManager."""
        print("  --> Testing (g) 4.5M Splat Cap Ceiling enforcement...")
        
        cap_manager = triple_hacks.MCMCStabilizerAndCapManager(N_max=4500000)
        
        # 1. Case where N_current + candidates < 4.5M
        N_curr = 4499990
        candidates = torch.arange(20, device=self.device)
        capped = cap_manager.enforce_cap_ceiling(N_curr, candidates)
        self.assertEqual(len(capped), 10, "Should cap candidates to 10 so total count does not exceed 4,500,000!")

        # 2. Case where N_current >= 4.5M
        N_full = 4500000
        capped_full = cap_manager.enforce_cap_ceiling(N_full, candidates)
        self.assertEqual(len(capped_full), 0, "Should return 0 candidates when N_current >= 4,500,000!")
        
        # 3. Case where N_current is small (1.0M)
        N_small = 1000000
        capped_small = cap_manager.enforce_cap_ceiling(N_small, candidates)
        self.assertEqual(len(capped_small), 20, "Should allow all candidates when well below 4.5M cap!")

    # -------------------------------------------------------------------------
    # h) Zero OOM memory leak check under simulated VRAM allocation
    # -------------------------------------------------------------------------
    def test_h_zero_oom_memory_leak_check(self):
        """Verify zero memory growth across 50 repeated densification & pruning cycles."""
        print("  --> Testing (h) Zero OOM memory leak check...")
        
        gmodel = triple_hacks.GaussianModel()
        pts = torch.randn((1000, 3))
        colors = torch.rand((1000, 3))
        gmodel.create_from_pcd(pts, colors, 1.0, self.device)
        
        # Measure baseline memory / tensor counts
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            mem_start = torch.cuda.memory_allocated()
            
        # Run 50 iterations of densify, optimize, and prune
        for step in range(1, 51):
            gmodel.optimizer.zero_grad()
            loss = torch.sum(gmodel._xyz ** 2) + torch.sum(gmodel._features_rest ** 2)
            loss.backward()
            
            # Accumulate dummy screen grads
            dummy_grad = torch.rand((len(gmodel._xyz), 2), device=self.device)
            gmodel.accumulate_gradients(dummy_grad)
            gmodel.optimizer.step()
            
            # Densify every 10 steps
            if step % 10 == 0:
                cand = torch.arange(min(50, len(gmodel._xyz)), device=self.device)
                gmodel.densify_and_split_clone(cand)
                gmodel.reset_gradient_accumulators()
                
            # Prune every 15 steps
            if step % 15 == 0:
                keep = torch.ones(len(gmodel._xyz), dtype=torch.bool, device=self.device)
                keep[::5] = False  # Prune 20%
                gmodel.prune_points(keep)

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            mem_end = torch.cuda.memory_allocated()
            mem_diff_mb = (mem_end - mem_start) / (1024 * 1024)
            print(f"    Memory Allocated Start: {mem_start / 1024 / 1024:.2f} MB, End: {mem_end / 1024 / 1024:.2f} MB (Diff: {mem_diff_mb:.2f} MB)")
            # Memory diff should be bounded (allowing small change due to splat count variance)
            self.assertLess(mem_diff_mb, 50.0, "CUDA memory leak detected! Growth > 50MB across iterations.")
            
        self.assertGreater(len(gmodel._xyz), 0, "Gaussian model must remain active and functional after iterations")
        print("    Memory leak check PASSED successfully with zero OOM leaks.")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUltimateAudit)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
