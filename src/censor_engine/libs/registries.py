from censor_engine.models.lib_models.registry import Registry

EffectRegistry = Registry("censor_engine.libs.effects")
MaskRegistry = Registry("censor_engine.libs.masks")
AIModelRegistry = Registry("censor_engine.libs.detectors.ai_models")
DetectorRegistry = Registry("censor_engine.libs.detectors.detector_interfaces")
