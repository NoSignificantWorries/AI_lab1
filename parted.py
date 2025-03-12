import os
import shutil
import math

path = "/home/dmitry/.cache/kagglehub/datasets/alexattia/the-simpsons-characters-dataset/versions/4/simpsons_dataset"

dataset_dir = "dataset"
os.mkdir(dataset_dir)

for group in ["train", "valid"]:
    os.mkdir(f"{dataset_dir}/{group}")
    for character in os.listdir(path):
        os.mkdir(f"{dataset_dir}/{group}/{character}")
        characters = os.listdir(f"{path}/{character}")
        if group == "train":
            g_charatcters = characters[:math.floor(0.8 * len(characters)) + 1]
        else:
            g_charatcters = characters[math.floor(0.8 * len(characters)) + 1:]
        for image in g_charatcters:
            try:
                shutil.copy2(f"{path}/{character}/{image}", f"{dataset_dir}/{group}/{character}/{image}")
            except BaseException as error:
                print(f"{character}/{image} ignored")

data = ['otto_mann', 'mayor_quimby', 'carl_carlson', 'sideshow_bob', 'patty_bouvier', 'troy_mcclure', 'gil',
        'selma_bouvier', 'waylon_smithers', 'agnes_skinner', 'marge_simpson', 'moe_szyslak', 'cletus_spuckler', 'principal_skinner',
        'edna_krabappel', 'rainier_wolfcastle', 'martin_prince', 'charles_montgomery_burns', 'lisa_simpson', 'lionel_hutz', 'lenny_leonard',
        'sideshow_mel', 'ralph_wiggum', 'professor_john_frink', 'milhouse_van_houten', 'bart_simpson', 'barney_gumble', 'ned_flanders',
        'snake_jailbird', 'kent_brockman', 'comic_book_guy', 'chief_wiggum', 'miss_hoover', 'krusty_the_clown', 'apu_nahasapeemapetilon',
        'nelson_muntz', 'maggie_simpson', 'fat_tony', 'homer_simpson', 'abraham_grampa_simpson', 'disco_stu', 'groundskeeper_willie']
