"""
iot_simulator.py — ECG Input Simulation & IoT Interface
=========================================================
Provides a modular interface for ECG input that can switch between:
  1. Manual user input (current)
  2. Simulated IoT garment data (demo)
  3. Real IoT garment data (future integration)

Designed for easy extension when a real wearable ECG device is connected.
"""

import random


class ECGSource:
    """Base class for ECG data sources."""

    def get_ecg_reading(self) -> dict:
        """
        Return ECG-related readings.

        Returns
        -------
        dict with keys:
            - restecg : str — "normal", "lv hypertrophy", or "st-t abnormality"
            - thalch  : float — maximum heart rate
            - exang   : bool — exercise-induced angina
            - oldpeak : float — ST depression
        """
        raise NotImplementedError("Subclasses must implement get_ecg_reading()")


class ManualECGInput(ECGSource):
    """ECG values provided manually by the user via the web form."""

    def __init__(self, restecg="normal", thalch=150, exang=False, oldpeak=0.0):
        self.restecg = restecg
        self.thalch = thalch
        self.exang = exang
        self.oldpeak = oldpeak

    def get_ecg_reading(self) -> dict:
        return {
            "restecg": self.restecg,
            "thalch": self.thalch,
            "exang": self.exang,
            "oldpeak": self.oldpeak,
        }


class SimulatedIoTGarment(ECGSource):
    """
    Simulates ECG readings from an IoT wearable garment.
    Useful for demos and testing without real hardware.
    """

    def __init__(self, profile="normal"):
        """
        Parameters
        ----------
        profile : str
            "normal"   — healthy baseline
            "abnormal" — elevated risk indicators
            "random"   — randomised values
        """
        self.profile = profile

    def get_ecg_reading(self) -> dict:
        if self.profile == "normal":
            return {
                "restecg": "normal",
                "thalch": random.randint(140, 180),
                "exang": False,
                "oldpeak": round(random.uniform(0, 0.5), 1),
            }
        elif self.profile == "abnormal":
            return {
                "restecg": random.choice(["lv hypertrophy", "st-t abnormality"]),
                "thalch": random.randint(70, 120),
                "exang": True,
                "oldpeak": round(random.uniform(1.5, 4.0), 1),
            }
        else:  # random
            return {
                "restecg": random.choice(["normal", "lv hypertrophy", "st-t abnormality"]),
                "thalch": random.randint(80, 190),
                "exang": random.choice([True, False]),
                "oldpeak": round(random.uniform(0, 4.5), 1),
            }


class RealIoTGarment(ECGSource):
    """
    Placeholder for real IoT garment integration.
    When a physical device SDK is available, implement the
    `get_ecg_reading()` method to pull live data.
    """

    def __init__(self, device_id=None, connection_url=None):
        self.device_id = device_id
        self.connection_url = connection_url
        # TODO: initialise BLE / MQTT / HTTP connection to the device

    def get_ecg_reading(self) -> dict:
        # TODO: Replace with actual device communication
        raise NotImplementedError(
            "Real IoT garment integration is not yet implemented. "
            "Connect your wearable device SDK here."
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def get_ecg_input(source: str = "manual", **kwargs) -> dict:
    """
    Factory function to get ECG data from the specified source.

    Parameters
    ----------
    source : str
        "manual"    — use kwargs as manual input
        "simulated" — use the IoT simulator (profile in kwargs)
        "iot"       — use real IoT device (not yet implemented)

    Returns
    -------
    dict with ECG-related values
    """
    if source == "manual":
        return ManualECGInput(
            restecg=kwargs.get("restecg", "normal"),
            thalch=kwargs.get("thalch", 150),
            exang=kwargs.get("exang", False),
            oldpeak=kwargs.get("oldpeak", 0.0),
        ).get_ecg_reading()

    elif source == "simulated":
        profile = kwargs.get("profile", "random")
        return SimulatedIoTGarment(profile=profile).get_ecg_reading()

    elif source == "iot":
        return RealIoTGarment(
            device_id=kwargs.get("device_id"),
            connection_url=kwargs.get("connection_url"),
        ).get_ecg_reading()

    else:
        raise ValueError(f"Unknown ECG source: {source}")
