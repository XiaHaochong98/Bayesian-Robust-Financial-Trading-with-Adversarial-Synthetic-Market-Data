from mmengine.registry import Registry

DATASET = Registry('data', locations=['module.data'])
PROMPT = Registry('prompt', locations=['module.prompt'])
AGENT = Registry('agent', locations=['module.agent'])
PROVIDER = Registry('provider', locations=['module.provider'])
DOWNLOADER = Registry('downloader', locations=['module.downloader'])
PROCESSOR = Registry('processor', locations=['module.processor'])
ENVIRONMENT = Registry('environment', locations=['module.environment'])
MEMORY = Registry('memory', locations=['module.memory'])
PLOTS = Registry('plot', locations=['module.plots'])