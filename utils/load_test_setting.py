from .settings import *
import os

filename = "test_settings.json"
settings = JsonConfig()
settings.load_json_file(filename)

with_diffusion = settings.with_diffusion

dataset_path = settings.dataset_path
batch_size = 1
model_epoch = settings.model_epoch
strength_factor = settings.strength_factor
lr = 1e-3
H, W, message_length = settings.H, settings.W, settings.message_length
noise_layers = settings.noise_layers

result_folder = "results/" + settings.result_folder
test_base = "/test_"
for layer in settings.noise_layers:
	test_base += layer + "_"
test_param = result_folder + test_base + "s{}_params.json".format(strength_factor)
test_log = result_folder + test_base + "s{}_log.txt".format(strength_factor)
os.makedirs(os.path.dirname(test_param), exist_ok=True)
with open(test_param, "w") as file:
	content = ""
	for item in settings.get_items():
		content += item[0] + " = " + str(item[1]) + "\n"
	print(content)

	with open(filename, "r") as setting_file:
		content = setting_file.read()
		file.write(content)

os.makedirs(os.path.dirname(test_log), exist_ok=True)
with open(test_log, "w") as file:
	file.write("")
