from models import modelserver
import settings
import sys
sys.path.append(".")

modelserver.initialize_models(model_path=settings.model_path)
modelserver.run()
