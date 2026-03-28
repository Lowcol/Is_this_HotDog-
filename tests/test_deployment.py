import unittest
from src.pipelines.deploy_model import deployment_trigger, DeploymentTriggerConfig

class TestDeploymentTrigger(unittest.TestCase):
    def setUp(self):
        # We require at least 80% precision and 75% recall to deploy
        self.config = DeploymentTriggerConfig(
            min_precision=0.80,
            min_recall=0.75
        )

    def test_deployment_meets_thresholds(self):
        """Test trigger when both metrics are comfortably above minimum."""
        result = deployment_trigger(
            precision=0.85,
            recall=0.80,
            config=self.config
        )
        self.assertTrue(result, "Should deploy when both metrics exceed thresholds")

    def test_deployment_fails_precision(self):
        """Test trigger fails when precision is too low."""
        result = deployment_trigger(
            precision=0.70, # Below 0.80
            recall=0.80,
            config=self.config
        )
        self.assertFalse(result, "Should NOT deploy when precision is below threshold")

    def test_deployment_fails_recall(self):
        """Test trigger fails when recall is too low."""
        result = deployment_trigger(
            precision=0.85,
            recall=0.70, # Below 0.75
            config=self.config
        )
        self.assertFalse(result, "Should NOT deploy when recall is below threshold")

    def test_deployment_fails_both(self):
        """Test trigger fails when both metrics are too low."""
        result = deployment_trigger(
            precision=0.50,
            recall=0.60,
            config=self.config
        )
        self.assertFalse(result, "Should NOT deploy when both metrics are failing")

if __name__ == '__main__':
    unittest.main()
