import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kfnoise import (fit_constrained_mle, fit_mle, half_diff_variance,  # noqa: E402
                     kalman_filter, neg_log_likelihood, roll_noise_variance,
                     rts_smoother, steady_state_gain)
from kfnoise import metrics as M  # noqa: E402
from kfnoise.data import SAMPLE_CSV, simulate_local_level  # noqa: E402


@pytest.fixture(scope="module")
def synth():
    return simulate_local_level(n=5000, q_std=0.5, r_std=2.0, seed=7)


def test_gain_converges_to_closed_form(synth):
    _, obs = synth
    res = kalman_filter(obs, 0.25, 4.0)
    assert res.K[-1] == pytest.approx(steady_state_gain(0.25, 4.0), rel=1e-8)


def test_gain_limits():
    assert steady_state_gain(1e6, 1.0) > 0.999        # trust the data
    assert steady_state_gain(1e-6, 1.0) < 0.002       # trust the model
    assert steady_state_gain(1.0, 0.0) == 1.0


def test_mle_recovers_parameters(synth):
    _, obs = synth
    f = fit_mle(obs)
    assert f.Q == pytest.approx(0.25, rel=0.25)
    assert f.R == pytest.approx(4.0, rel=0.10)


def test_mle_beats_true_params_likelihood(synth):
    _, obs = synth
    f = fit_mle(obs)
    assert f.nll <= neg_log_likelihood(obs, 0.25, 4.0) + 1e-6


def test_roll_estimator_unbiased_but_halfdiff_biased(synth):
    _, obs = synth
    assert roll_noise_variance(obs) == pytest.approx(4.0, rel=0.15)
    # Var(dz)/2 = R + Q/2 in expectation
    assert half_diff_variance(obs) == pytest.approx(4.0 + 0.125, rel=0.05)


def test_constrained_mle_fixed_r(synth):
    _, obs = synth
    f = fit_constrained_mle(obs, R=4.0)
    assert f.R == 4.0
    assert f.Q == pytest.approx(0.25, rel=0.25)


def test_filtering_reduces_error(synth):
    truth, obs = synth
    res = kalman_filter(obs, 0.25, 4.0)
    xs, _ = rts_smoother(res, 0.25)
    rmse = lambda a: np.sqrt(np.mean((a - truth) ** 2))  # noqa: E731
    assert rmse(xs) < rmse(res.x_filt) < rmse(obs)


def test_innovations_white_when_model_correct(synth):
    _, obs = synth
    res = kalman_filter(obs, 0.25, 4.0)
    _, p = M.ljung_box(res.innov[50:], 10)
    assert p > 0.01


def test_matches_pykalman_filter():
    pk = pytest.importorskip("pykalman")
    _, obs = simulate_local_level(n=300, seed=3)
    P0 = 10.0
    # pykalman applies the first measurement update without a time step (no +Q
    # at t=0), so start our recursion from P0 - Q to make the two identical.
    ours = kalman_filter(obs, 0.3, 2.0, x0=obs[0], P0=P0 - 0.3)
    kf = pk.KalmanFilter(transition_matrices=[[1.0]], observation_matrices=[[1.0]],
                         transition_covariance=[[0.3]], observation_covariance=[[2.0]],
                         initial_state_mean=[obs[0]], initial_state_covariance=[[P0]])
    means, _ = kf.filter(obs.reshape(-1, 1))
    np.testing.assert_allclose(ours.x_filt, means.ravel(), rtol=0, atol=1e-9)


def test_sample_data_present():
    assert SAMPLE_CSV.exists()
