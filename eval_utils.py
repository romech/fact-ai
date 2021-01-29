classification_path = "classification"
angle_estimation_path = "angle_estimation"
inv_attack1_path = "inv_attack1"
inv_attack2_path = "inv_attack2"

additional_prototype_file = "inference_attacks/resnet_56_additional_prototype/best-epoch=139-val_acc=0.7808.ckpt"
additional_inv_model_file = "inference_attacks/resnet_56_additional_prototype_inversion_model/best-epoch=12-val_loss=0.0931.ckpt"

complex_prototype_file = "inference_attacks/resnet_56_complex_prototype/best-epoch=195-val_acc=0.7249.ckpt"
angle_estimator_file = "inference_attacks/resnet_56_complex_prototype_angle_estimation/best-epoch=47-val_loss=0.1760.ckpt"
complex_inv_model_file = "inference_attacks/resnet_56_complex_prototype_angle_feature_inversion/best-epoch=14-val_loss=0.2859.ckpt"

inf_attack_path = "inference_attacks"
inf_attack2_additional_path = "resnet_56_attacker_on_additional_prototype_infattack2/last.ckpt"
inf_attack2_complex_path = "resnet_56_attacker_on_complex_prototype_infattack2/last.ckpt"

inf_attack3_additional_path = "resnet_56_attacker_on_additional_prototype_infattack3/last.ckpt"
inf_attack3_complex_path = "resnet_56_attacker_on_complex_prototype_infattack3/last.ckpt"

resnet_classification_experiments = {
    "ResNet-20-a": {
        "Original": "resnet_20_alpha_original_cif10/best-epoch=196-val_acc=0.9223.ckpt",
        "Additional": "resnet_20_alpha_additional_layers_cif10/best-epoch=106-val_acc=0.9094.ckpt",
        "Complex": "resnet_20_alpha_cif10_complex/best-epoch=168-val_acc=0.8838.ckpt"
    }, "ResNet-20-b": {
        "Original": "resnet_20_beta_original_cif10/best-epoch=198-val_acc=0.9197.ckpt",
        "Additional": "resnet_20_beta_additional_layers_cif10/best-epoch=169-val_acc=0.9170.ckpt",
        "Complex": "resnet_20_beta_cif10_complex/best-epoch=155-val_acc=0.8720.ckpt"
    }, "ResNet-32-a": {
        "Original": "resnet_32_alpha_original_cif10/best-epoch=188-val_acc=0.9273.ckpt",
        "Additional": "resnet_32_alpha_additional_layers_cif10/best-epoch=187-val_acc=0.9105.ckpt",
        "Complex": "resnet_32_alpha_cif10_complex/best-epoch=105-val_acc=0.8910.ckpt"
    }, "ResNet-32-b": {
        "Original": "resnet_32_beta_original_cif10/best-epoch=193-val_acc=0.9278.ckpt",
        "Additional": "resnet_32_beta_additional_layers_cif10/best-epoch=175-val_acc=0.9112.ckpt",
        "Complex": "resnet_32_beta_cif10_complex/best-epoch=107-val_acc=0.8957.ckpt"
    }, "ResNet-44-a": {
        "Original": "resnet_44_alpha_original_cif10/best-epoch=167-val_acc=0.9343.ckpt",
        "Additional": "resnet_44_alpha_additional_layers_cif10/best-epoch=166-val_acc=0.9068.ckpt",
        "Complex": "resnet_44_alpha_cif10_complex/best-epoch=103-val_acc=0.8997.ckpt"
    }, "ResNet-44-b": {
        "Original": "resnet_44_beta_original_cif10/best-epoch=152-val_acc=0.9269.ckpt",
        "Additional": "resnet_44_beta_additional_layers_cif10/best-epoch=118-val_acc=0.9116.ckpt",
        "Complex": "resnet_44_beta_cif10_complex/best-epoch=120-val_acc=0.9009.ckpt"
    }, "ResNet-56-a": {
        "Original": "resnet_56_alpha_original_cif10/best-epoch=169-val_acc=0.9281.ckpt",
        "Additional": "resnet_56_alpha_additional_layers_cif10/best-epoch=188-val_acc=0.9155.ckpt",
        "Complex": "resnet_56_alpha_cif10_complex/best-epoch=114-val_acc=0.9009.ckpt"
    }, "ResNet-56-b": {
        "Original": "resnet_56_beta_original_cif10/best-epoch=194-val_acc=0.9311.ckpt",
        "Additional": "resnet_56_beta_additional_layers_cif10/best-epoch=176-val_acc=0.9173.ckpt",
        "Complex": "resnet_56_beta_cif10_complex/best-epoch=193-val_acc=0.9021.ckpt"
    }
}

mixed_classification_experiments = {
    ("LeNet", "CIFAR-10"): {
        "Original": "lenet_cif10_original/best-epoch=154-val_acc=0.7391.ckpt",
        "Additional": "lenet_cif10_additional/best-epoch=176-val_acc=0.7362.ckpt",
        "Noisy (gamma = 0.2)": "lenet_cif10_noisy-0_2/best-epoch=196-val_acc=0.7389.ckpt",
        "Noisy (gamma = 0.5)": "lenet_cif10_noisy-0_5/best-epoch=174-val_acc=0.7122.ckpt",
        "Noisy (gamma = 1.0)": "lenet_cif10_noisy-1_0/best-epoch=184-val_acc=0.6543.ckpt",
        "Complex": "lenet_cif10_complex/best-epoch=186-val_acc=0.6717.ckpt"
    },
    ("LeNet", "CIFAR-100"): {
        "Original": "lenet_cif100_original/best-epoch=191-val_acc=0.3911.ckpt",
        "Additional": "lenet_cif100_additional/best-epoch=163-val_acc=0.3966.ckpt",
        "Noisy (gamma = 0.2)": "lenet_cif100_noisy-0_2/best-epoch=160-val_acc=0.3881.ckpt",
        "Noisy (gamma = 0.5)": "lenet_cif100_noisy-0_5/best-epoch=154-val_acc=0.3557.ckpt",
        "Noisy (gamma = 1.0)": "lenet_cif100_noisy-1_0/best-epoch=177-val_acc=0.3357.ckpt",
        "Complex": "lenet_cif100_complex/best-epoch=194-val_acc=0.2623.ckpt"
    },
    ("ResNet-56-a", "CIFAR-100"): {
        "Original": "resnet56_cif100_original/best-epoch=109-val_acc=0.6780.ckpt",
        "Additional": "resnet56_cif100_additional/best-epoch=108-val_acc=0.6754.ckpt",
        "Noisy (gamma = 0.2)": "resnet56_cif100_noisy-0_2/best-epoch=172-val_acc=0.6728.ckpt",
        "Noisy (gamma = 0.5)": "resnet56_cif100_noisy-0_5/best-epoch=107-val_acc=0.6675.ckpt",
        "Noisy (gamma = 1.0)": "resnet56_cif100_noisy-1_0/best-epoch=127-val_acc=0.6622.ckpt",
        "Complex": "resnet56_cif100_complex/best-epoch=103-val_acc=0.6452.ckpt"
    }
}

angle_estimation_experiments = {
    ("ResNet-20-a", "CIFAR-10"): "resnet_20_alpha_cif10/best-epoch=46-val_loss=0.0865.ckpt",
    ("ResNet-20-b", "CIFAR-10"): "resnet_20_beta_cif10/best-epoch=47-val_loss=0.0945.ckpt",
    ("ResNet-32-a", "CIFAR-10"): "resnet_32_alpha_cif10/best-epoch=46-val_loss=0.0900.ckpt",
    ("ResNet-32-b", "CIFAR-10"): "resnet_32_beta_cif10/best-epoch=41-val_loss=0.0957.ckpt",
    ("ResNet-44-a", "CIFAR-10"): "resnet_44_alpha_cif10/best-epoch=40-val_loss=0.0848.ckpt",
    ("ResNet-44-b", "CIFAR-10"): "resnet_44_beta_cif10/best-epoch=49-val_loss=0.0959.ckpt",
    ("ResNet-56-a", "CIFAR-10"): "resnet_56_alpha_cif10/best-epoch=37-val_loss=0.0946.ckpt",
    ("ResNet-56-b", "CIFAR-10"): "resnet_56_beta_cif10/best-epoch=44-val_loss=0.0896.ckpt",
    ("LeNet", "CIFAR-10"): "lenet_cif10/best-epoch=46-val_loss=0.2446.ckpt",
    ("LeNet", "CIFAR-100"): "lenet_cif100/best-epoch=47-val_loss=0.3164.ckpt",
    ("ResNet-56-a", "CIFAR-100"): "resnet56_cif100/best-epoch=47-val_loss=0.0862.ckpt",
}

resnet_inversion_experiments = {
    "ResNet-20-a": {
        "Original": "resnet_20_alpha_original_cif10/best-epoch=14-val_loss=0.0750.ckpt",
        "Additional": "resnet_20_alpha_additional_layers_cif10/best-epoch=13-val_loss=0.1003.ckpt",
        "Complex dec(a*)": "resnet_20_alpha_cif10/best-epoch=14-val_loss=0.1920.ckpt",
        "Complex dec(x)": "resnet_20_alpha_cif10_complex/best-epoch=14-val_loss=0.2388.ckpt"
    }, "ResNet-20-b": {
        "Original": "resnet_20_beta_original_cif10/best-epoch=13-val_loss=0.0807.ckpt",
        "Additional": "resnet_20_beta_additional_layers_cif10/best-epoch=13-val_loss=0.1009.ckpt",
        "Complex dec(a*)": "resnet_20_beta_cif10/best-epoch=14-val_loss=0.2288.ckpt",
        "Complex dec(x)": "resnet_20_beta_cif10_complex/best-epoch=14-val_loss=0.2860.ckpt"
    }, "ResNet-32-a": {
        "Original": "resnet_32_alpha_original_cif10/best-epoch=13-val_loss=0.0927.ckpt",
        "Additional": "resnet_32_alpha_additional_layers_cif10/best-epoch=14-val_loss=0.1872.ckpt",
        "Complex dec(a*)": "resnet_32_alpha_cif10/best-epoch=13-val_loss=0.1946.ckpt",
        "Complex dec(x)": "resnet_32_alpha_cif10_complex/best-epoch=14-val_loss=0.2538.ckpt"
    }, "ResNet-32-b": {
        "Original": "resnet_32_beta_original_cif10/best-epoch=14-val_loss=0.0748.ckpt",
        "Additional": "resnet_32_beta_additional_layers_cif10/best-epoch=14-val_loss=0.2201.ckpt",
        "Complex dec(a*)": "resnet_32_beta_cif10/best-epoch=14-val_loss=0.2056.ckpt",
        "Complex dec(x)": "resnet_32_beta_cif10_complex/best-epoch=14-val_loss=0.2677.ckpt"
    }, "ResNet-44-a": {
        "Original": "resnet_44_alpha_original_cif10/best-epoch=11-val_loss=0.0735.ckpt",
        "Additional": "resnet_44_alpha_additional_layers_cif10/best-epoch=13-val_loss=0.2132.ckpt",
        "Complex dec(a*)": "resnet_44_alpha_cif10/best-epoch=14-val_loss=0.2235.ckpt",
        "Complex dec(x)": "resnet_44_alpha_cif10_complex/best-epoch=14-val_loss=0.2883.ckpt"
    }, "ResNet-44-b": {
        "Original": "resnet_44_beta_original_cif10/best-epoch=11-val_loss=0.0642.ckpt",
        "Additional": "resnet_44_beta_additional_layers_cif10/best-epoch=10-val_loss=0.2192.ckpt",
        "Complex dec(a*)": "resnet_44_beta_cif10/best-epoch=14-val_loss=0.2280.ckpt",
        "Complex dec(x)": "resnet_44_beta_cif10_complex/best-epoch=13-val_loss=0.2991.ckpt"
    }, "ResNet-56-a": {
        "Original": "resnet_56_alpha_original_cif10/best-epoch=14-val_loss=0.0581.ckpt",
        "Additional": "resnet_56_alpha_additional_layers_cif10/best-epoch=10-val_loss=0.0866.ckpt",
        "Complex dec(a*)": "resnet_56_alpha_cif10/best-epoch=12-val_loss=0.2126.ckpt",
        "Complex dec(x)": "resnet_56_alpha_cif10_complex/best-epoch=12-val_loss=0.2840.ckpt"
    }, "ResNet-56-b": {
        "Original": "resnet_56_beta_original_cif10/best-epoch=14-val_loss=0.0582.ckpt",
        "Additional": "resnet_56_beta_additional_layers_cif10/best-epoch=12-val_loss=0.0935.ckpt",
        "Complex dec(a*)": "resnet_56_beta_cif10/best-epoch=13-val_loss=0.2488.ckpt",
        "Complex dec(x)": "resnet_56_beta_cif10_complex/best-epoch=12-val_loss=0.3413.ckpt"
    }
}

mixed_inversion_experiments = {
    ("LeNet", "CIFAR-10"): {
        "Original": "lenet_cif10_original/best-epoch=13-val_loss=0.2070.ckpt",
        "Additional": "lenet_cif100_additional/best-epoch=11-val_loss=0.2374.ckpt",
        "Noisy (gamma = 0.2)": "lenet_cif10_noisy-0_2/best-epoch=13-val_loss=0.2102.ckpt",
        "Noisy (gamma = 0.5)": "lenet_cif10_noisy-0_5/best-epoch=11-val_loss=0.2410.ckpt",
        "Noisy (gamma = 1.0)": "lenet_cif10_noisy-1_0/best-epoch=13-val_loss=0.3667.ckpt",
        "Complex dec(a*)": "lenet_cif10/best-epoch=13-val_loss=0.4297.ckpt",
        "Complex dec(x)": "lenet_cif10_complex/best-epoch=14-val_loss=0.4409.ckpt"
    },
    ("LeNet", "CIFAR-100"): {
        "Original": "lenet_cif100_original/best-epoch=13-val_loss=0.1698.ckpt",
        "Additional": "lenet_cif100_additional/best-epoch=11-val_loss=0.2374.ckpt",
        "Noisy (gamma = 0.2)": "lenet_cif100_noisy-0_2/best-epoch=14-val_loss=0.2012.ckpt",
        "Noisy (gamma = 0.5)": "lenet_cif100_noisy-0_5/best-epoch=13-val_loss=0.2049.ckpt",
        "Noisy (gamma = 1.0)": "lenet_cif100_noisy-1_0/best-epoch=6-val_loss=0.2537.ckpt",
        "Complex dec(a*)": "lenet_cif100/best-epoch=9-val_loss=0.2819.ckpt",
        "Complex dec(x)": "lenet_cif100_complex/best-epoch=14-val_loss=0.2496.ckpt"
    },
    ("ResNet-56-a", "CIFAR-100"): {
        "Original": "resnet56_cif100_original/best-epoch=10-val_loss=0.0830.ckpt",
        "Additional": "resnet56_cif100_additional/best-epoch=14-val_loss=0.0955.ckpt",
        "Noisy (gamma = 0.2)": "resnet56_cif100_noisy-0_2/best-epoch=10-val_loss=0.1007.ckpt",
        "Noisy (gamma = 0.5)": "resnet56_cif100_noisy-0_5/best-epoch=11-val_loss=0.1277.ckpt",
        "Noisy (gamma = 1.0)": "resnet56_cif100_noisy-1_0/best-epoch=11-val_loss=0.1490.ckpt",
        "Complex dec(a*)": "resnet56_cif100/best-epoch=11-val_loss=0.1535.ckpt",
        "Complex dec(x)": "resnet56_cif100_complex/best-epoch=14-val_loss=0.2143.ckpt"
    }
}
