import json
import os
import random
import sys
import time
from copy import copy
from pprint import pprint

import imageio
import numpy
import requests

try_on_image_template = "https://media.tryoncloth.com/generated_model_image/{model_file}.png"


STYLE_SPACE_APP_HEADERS = {
    "User-Agent": "FashionCamera/1.08 (com.tryoncloth.stylespace; build:0; iOS 12.5.1) Alamofire/5.2.2",
}


def style_space_login():
    headers = copy(STYLE_SPACE_APP_HEADERS)
    r = requests.post(
        "https://app.tryoncloth.com/sign_in",
        headers=headers,
        json={
            "credential": "anuf10h",
            "credential_type": "password",
            "identifier": "stylespace@dsumin.ru",
            "identifier_type": "email",
        },
    )
    if r.status_code == 200:
        # Returned sample:
        # {
        #     "key": "2q132ZuX",
        #     "onboarding_status": {
        #         "has_personal_model": false,
        #         "has_personal_preference": false,
        #         "has_profile_photo": true,
        #         "has_selected_sites": false
        #     },
        #     "refresh_token": "pbkdf2:sha256:2044$5knuIvqW$35d5a7a8d8d68e920be87d01f2405708c5ad379e1fabc6b755c5...",
        #     "session_token": "pbkdf2:sha256:2044$FioPnsGQ$89146248c4d8df18ddd3d1ce4e722a6a4f63caaf9eab4661c90c...",
        #     "success": true,
        #     "user_id": "KgXKWJfl3beae18118a14b8b29ccac1429846233b07a0f80c47443e3d80a9b14b5e2c0be"
        # }
        return r.json()
    else:
        print(r.status_code, r.content)
        pprint(r.json())
        raise RuntimeError("Failed to login to StyleSpace")


def make_session(user_id, refresh_token):
    headers = copy(STYLE_SPACE_APP_HEADERS)
    cred = {
        "credential": refresh_token,
        "credential_type": "refresh_token",
        "user_id": user_id,
    }

    r = requests.post("https://app.tryoncloth.com/start_session", headers=headers, json=cred)
    if r.status_code == 200:
        return r.json()
    else:
        print(r.status_code, r.json())


def make_auth_headers():
    login_data = style_space_login()
    session = make_session(login_data["user_id"], login_data["refresh_token"])

    auth_headers = copy(STYLE_SPACE_APP_HEADERS)
    auth_headers.update(
        {
            "session_token": session["session_token"],
            "user_id": login_data["user_id"],
        }
    )
    return auth_headers


def process_one_model(data, images_dir):
    gender = data["gender"]
    # pprint(random_popular_look)
    model_id = data["model_id"]
    model_file = data["model_file"]
    print(gender, model_file)
    model_image = imageio.imread(try_on_image_template.format(model_file=model_file))
    model_file_dir = os.path.join(images_dir, gender, model_id, model_file)
    os.makedirs(model_file_dir)
    imageio.imwrite(os.path.join(model_file_dir, "model_file.png"), model_image)
    for category, product_id in zip(data["product_categories"], data["product_ids"]):
        url = data["front_images"][product_id]
        item_image = imageio.imread(url)
        imageio.imwrite(os.path.join(model_file_dir, f"{category}.jpg"), item_image)
    with open(os.path.join(model_file_dir, "meta.json"), "w") as f:
        json.dump(data, f)


def main(scrape_gender, random_seed):
    random.seed(random_seed)

    auth_headers = make_auth_headers()

    products_collection_file = "../style_space_products.json"
    if os.path.exists(products_collection_file):
        with open(products_collection_file, "r") as f:
            products_collection = json.load(f)
    else:
        search_products_args = {
            "allow_multiple_in_variation": "",
            "in_stock": "",
            "max_price": "",
            "min_price": "",
            "page_size": "50",
            "search_text": "",
            "sort_by": "time",
            "tryon_enabled": "",
            "variation_id": "",
            "gender": None,
            "tryon_categories[]": None,
            "page": None,
        }

        products_collection = {}

        for gender in ("male", "female"):
            products_collection[gender] = {}
            for category in ("tops", "bottoms", "outerwear", "allbody"):
                products_collection[gender][category] = []
                page = 0
                while True:
                    search_products_args.update(
                        {
                            "gender": gender,
                            "tryon_categories[]": category,
                            "page": page,
                        }
                    )
                    print(gender, category, page)
                    search_products_args_query_string = "&".join([f"{k}={v}" for k, v in search_products_args.items()])
                    r = requests.get(
                        "https://app.tryoncloth.com/search_product?" + search_products_args_query_string,
                        headers=STYLE_SPACE_APP_HEADERS,
                    )
                    if r.status_code == 200:
                        products = r.json()["products"]
                        if products:
                            products_collection[gender][category].extend(products)
                            page += 1
                        else:
                            break
                    else:
                        print(r.status_code, r.json())

        with open("../style_space_products.json", "w") as f:
            json.dump(products_collection, f)

    for gender in ("male", "female"):
        for category in ("tops", "bottoms", "outerwear", "allbody"):
            print(gender, category, len(products_collection[gender][category]))

    return

    models_collection_file = "../style_space_models.json"
    if os.path.exists(models_collection_file):
        with open(models_collection_file, "r") as f:
            models_collection = json.load(f)
    else:
        recommended_model_list_args = {
            "gender": None,
            "page": 0,
            "page_size": 20,
        }
        models_collection = {}
        for gender in ("male", "female"):
            models_collection[gender] = set()
            page = 0
            while True:
                recommended_model_list_args.update({"gender": gender, "page": page})
                recommended_model_list_args_list = list(recommended_model_list_args.items()) + [
                    ("categories[]", "bottoms"),
                    ("categories[]", "tops"),
                ]
                print(recommended_model_list_args_list)
                recommended_model_list_args_query_string = "&".join(
                    [f"{k}={v}" for k, v in recommended_model_list_args_list]
                )
                r = requests.get(
                    "https://app.tryoncloth.com/get_recommended_model_list?" + recommended_model_list_args_query_string
                )
                if r.status_code == 200:
                    model_ids = r.json()["model_ids"]
                    if model_ids:
                        models_collection[gender].update(model_ids)
                        print(len(models_collection[gender]))
                        page += 1
                    else:
                        models_collection[gender] = list(models_collection[gender])
                        break
                else:
                    print(r.status_code, r.json())
        with open(models_collection_file, "w") as f:
            json.dump(models_collection, f)

    for gender in ("male", "female"):
        print(gender, len(models_collection[gender]))

    images_dir = "../style_space_images"
    os.makedirs(images_dir, exist_ok=True)

    gender = scrape_gender

    g_models = models_collection[gender]
    g_bottoms = [bottom_cloth["id"] for bottom_cloth in products_collection[gender]["bottoms"]]
    g_tops = [bottom_cloth["id"] for bottom_cloth in products_collection[gender]["tops"]]

    total_combinations = len(g_models) * len(g_bottoms) * len(g_tops)
    print(f"Scraping {gender} models. Total {total_combinations} combinations")

    while True:
        try:
            random_index = random.randint(0, total_combinations)

            model_idx, bottom_idx, top_idx = numpy.unravel_index(
                random_index, (len(g_models), len(g_bottoms), len(g_tops))
            )
            model_id = g_models[model_idx]
            bottom_cloth_id = g_bottoms[bottom_idx]
            top_cloth_id = g_tops[top_idx]

            r = requests.post(
                "https://app.tryoncloth.com/create_tryon_for_products",
                headers=auth_headers,
                json={
                    "categories": ["tops", "bottoms"],
                    "model_id": model_id,
                    "product_ids": [
                        top_cloth_id,
                        bottom_cloth_id,
                    ],
                },
            )
            if r.status_code == 200:
                # pprint(r.json())
                process_one_model(r.json()["look_metadata"], images_dir)
                # break
            elif r.status_code == 401:
                auth_headers.update(make_auth_headers())
            else:
                print(r.status_code)
                print(r.content)
                # print(r.json())
        except Exception as e:
            print(e)
            time.sleep(3)
            continue

    return

    # while True:
    #     try:
    #         r = requests.get("https://app.tryoncloth.com/get_a_random_popular_look", headers=headers)
    #         # gender = random.choice(["male", "female"])
    #         # print(gender)
    #         # r = requests.get(f"https://revery.tryoncloth.com/generate_a_random_outfit?gender={gender}")
    #         if r.status_code == 200:
    #             process_one_model(r.json()["model_meta"], images_dir)
    #         else:
    #             print(r.status_code, r.json())
    #     except Exception as e:
    #         print(e)
    #         time.sleep(3)
    #         continue


if __name__ == "__main__":
    gender = sys.argv[1]
    random_seed = sys.argv[2]
    main(gender, random_seed)
