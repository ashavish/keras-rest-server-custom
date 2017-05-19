from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D



base_model = ResNet50(weights='imagenet',include_top=False)
x = base_model.output
model_pool = GlobalAveragePooling2D()(x)
model = Model(input=base_model.input, output=model_pool)

model.save("resnet_model.h5")