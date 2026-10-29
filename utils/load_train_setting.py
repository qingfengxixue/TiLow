from .settings import *
import os

'''
params setting
'''

settings = JsonConfig()
settings.load_json_file("train_settings.json")

with_diffusion = settings.with_diffusion

project_name = settings.project_name
dataset_path = settings.dataset_path
epoch_number = settings.epoch_number
batch_size = settings.batch_size
train_continue = settings.train_continue
train_continue_path = settings.train_continue_path
train_continue_epoch = settings.train_continue_epoch
decoder_only_init_path = getattr(settings, "decoder_only_init_path", "")
lr = settings.lr
H, W, message_length = settings.H, settings.W, settings.message_length,
alpha = getattr(settings, "alpha", 0.15)
stage2_alpha_decay = getattr(settings, "stage2_alpha_decay", False)
stage2_alpha_start = getattr(settings, "stage2_alpha_start", alpha)
stage2_alpha_hold_end_epoch = getattr(settings, "stage2_alpha_hold_end_epoch", 20)
stage2_alpha_end = getattr(settings, "stage2_alpha_end", 0.05)
stage2_alpha_end_epoch = getattr(settings, "stage2_alpha_end_epoch", 100)
stage2_alpha_phase1_start_epoch = getattr(settings, "stage2_alpha_phase1_start_epoch", 8)
stage2_alpha_phase1_end_epoch = getattr(settings, "stage2_alpha_phase1_end_epoch", stage2_alpha_hold_end_epoch)
stage2_alpha_phase1_value = getattr(settings, "stage2_alpha_phase1_value", stage2_alpha_start)
stage2_alpha_phase2_end_epoch = getattr(settings, "stage2_alpha_phase2_end_epoch", 30)
stage2_alpha_phase2_end = getattr(settings, "stage2_alpha_phase2_end", 0.1)
stage2_alpha_phase3_end_epoch = getattr(settings, "stage2_alpha_phase3_end_epoch", stage2_alpha_end_epoch)
stage2_alpha_phase3_end = getattr(settings, "stage2_alpha_phase3_end", stage2_alpha_end)
stage2_alpha_hold_on_high_error = getattr(settings, "stage2_alpha_hold_on_high_error", True)
stage2_alpha_errorrate_threshold = getattr(settings, "stage2_alpha_errorrate_threshold", 0.05)
stage2_alpha_adaptive = getattr(settings, "stage2_alpha_adaptive", False)
stage2_alpha_adaptive_start = getattr(settings, "stage2_alpha_adaptive_start", 0.5)
stage2_alpha_adaptive_threshold = getattr(settings, "stage2_alpha_adaptive_threshold", 0.05)
stage2_alpha_adaptive_step_high = getattr(settings, "stage2_alpha_adaptive_step_high", 0.05)
stage2_alpha_adaptive_switch = getattr(settings, "stage2_alpha_adaptive_switch", 0.1)
stage2_alpha_adaptive_step_low = getattr(settings, "stage2_alpha_adaptive_step_low", 0.005)
stage2_alpha_adaptive_switch2 = getattr(settings, "stage2_alpha_adaptive_switch2", 0.05)
stage2_alpha_adaptive_step_min = getattr(settings, "stage2_alpha_adaptive_step_min", 0.001)
stage2_alpha_adaptive_min = getattr(settings, "stage2_alpha_adaptive_min", 0.01)
locator_alpha_decay = getattr(settings, "locator_alpha_decay", True)
locator_alpha_hold_end_epoch = getattr(settings, "locator_alpha_hold_end_epoch", 10)
locator_alpha_drop1_start_epoch = getattr(settings, "locator_alpha_drop1_start_epoch", 11)
locator_alpha_drop1_end_epoch = getattr(settings, "locator_alpha_drop1_end_epoch", 14)
locator_alpha_drop1_step = getattr(settings, "locator_alpha_drop1_step", 0.1)
locator_alpha_drop2_start_epoch = getattr(settings, "locator_alpha_drop2_start_epoch", 15)
locator_alpha_drop2_step = getattr(settings, "locator_alpha_drop2_step", 0.01)
locator_alpha_drop2_start_value = getattr(settings, "locator_alpha_drop2_start_value", 0.09)
locator_alpha_phase1_start_epoch = getattr(settings, "locator_alpha_phase1_start_epoch", 1)
locator_alpha_phase1_end_epoch = getattr(settings, "locator_alpha_phase1_end_epoch", 5)
locator_alpha_phase1_start = getattr(settings, "locator_alpha_phase1_start", 0.5)
locator_alpha_phase1_end = getattr(settings, "locator_alpha_phase1_end", 0.1)
locator_alpha_phase2_end_epoch = getattr(settings, "locator_alpha_phase2_end_epoch", 10)
locator_alpha_phase2_end = getattr(settings, "locator_alpha_phase2_end", 0.05)
locator_alpha_plateau_delta = getattr(settings, "locator_alpha_plateau_delta", 0.01)
locator_alpha_plateau_monitor_epochs = getattr(settings, "locator_alpha_plateau_monitor_epochs", 10)
locator_alpha_early_stop_alpha = getattr(settings, "locator_alpha_early_stop_alpha", 0.03)
locator_alpha_early_stop_enable = getattr(settings, "locator_alpha_early_stop_enable", True)
noise_layers = settings.noise_layers
gaussian_aug = getattr(settings, "gaussian_aug", False)
gaussian_aug_prob = getattr(settings, "gaussian_aug_prob", 0.5)
gaussian_sigma_min = getattr(settings, "gaussian_sigma_min", 0.0)
gaussian_sigma_max = getattr(settings, "gaussian_sigma_max", 0.05)
use_cropout_attack = getattr(settings, "use_cropout_attack", False)
cropout_height_ratio = getattr(settings, "cropout_height_ratio", 0.5)
cropout_width_ratio = getattr(settings, "cropout_width_ratio", 0.5)
quick_train = getattr(settings, "quick_train", False)
quick_epochs = getattr(settings, "quick_epochs", 2)
quick_steps = getattr(settings, "quick_steps", 20)
log_image_every = getattr(settings, "log_image_every", 50)
vote_crops = getattr(settings, "vote_crops", 4)
enable_epoch_test_stage = getattr(settings, "enable_epoch_test_stage", True)
debug_test_only = getattr(settings, "debug_test_only", False)
test_stage_steps = getattr(settings, "test_stage_steps", 50)
test_crop_size = getattr(settings, "test_crop_size", 384)
test_uncertain_low = getattr(settings, "test_uncertain_low", 0.45)
test_uncertain_high = getattr(settings, "test_uncertain_high", 0.55)
grad_clip_norm = getattr(settings, "grad_clip_norm", 0.0)
decoder_weight = getattr(settings, "decoder_weight", 1.0)
use_second_noise = getattr(settings, "use_second_noise", True)
use_vote_mechanism = getattr(settings, "use_vote_mechanism", True)
locator_pretrain = getattr(settings, "locator_pretrain", False)
fixed_message = getattr(settings, "fixed_message", False)
fixed_message_seed = getattr(settings, "fixed_message_seed", 2026)
single_tile_embed = getattr(settings, "single_tile_embed", False)
random_tile_position = getattr(settings, "random_tile_position", True)
locator_feature_supervision = getattr(settings, "locator_feature_supervision", False)
locator_feat_weight = getattr(settings, "locator_feat_weight", 1.0)
preprocess_before_batch = getattr(settings, "preprocess_before_batch", True)
two_stage_training = getattr(settings, "two_stage_training", True)
locator_stage_epochs = getattr(settings, "locator_stage_epochs", 20)
decoder_stage_epochs = getattr(settings, "decoder_stage_epochs", 80)
locator_converge_threshold = getattr(settings, "locator_converge_threshold", 0.95)
locator_converge_patience = getattr(settings, "locator_converge_patience", 3)
locator_crop_size = getattr(settings, "locator_crop_size", 128)
locator_threshold = getattr(settings, "locator_threshold", 0.5)
locator_invalid_weight = getattr(settings, "locator_invalid_weight", 0.2)

'''
file preparing
'''

full_project_name = project_name + "_m" + str(message_length)
for noise in noise_layers:
	full_project_name += "_" + noise
result_folder = "results/run/"
os.makedirs(result_folder + "images/", exist_ok=True)
os.makedirs(result_folder + "models/", exist_ok=True)

with open(result_folder + "/train_params.txt", "w") as file:
	content = ""
	for item in settings.get_items():
		content += item[0] + " = " + str(item[1]) + "\n"

	print(content)
	file.write(content)

with open(result_folder + "/train_log.txt", "w") as file:
	file.write("")
with open(result_folder + "/val_log.txt", "w") as file:
	file.write("")
with open(result_folder + "/test_log.txt", "w") as file:
	file.write("")
