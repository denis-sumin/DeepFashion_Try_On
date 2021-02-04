import json
import os
from pprint import pprint
from time import sleep, time

import requests

STYLE_SPACE_APP_HEADERS = {
    "User-Agent": "FashionCamera/1.08 (com.tryoncloth.stylespace; build:0; iOS 12.5.1) Alamofire/5.2.2",
}

PRODUCTS_FILE = "../style_space_all_products.json"
PRODUCTS_DIR = "style_space_all_products"


def fetch_images(item):
    variation_id, item = item
    # from pprint import pprint
    # pprint(item)
    # break
    category = str(item.get("tryon", {}).get("category", "__no_tryon"))
    os.makedirs(os.path.join(PRODUCTS_DIR, category, variation_id), exist_ok=True)
    for variation in item["media_metadata"]:
        dst_dir = os.path.join(PRODUCTS_DIR, category, variation_id, variation["id"])
        os.makedirs(dst_dir, exist_ok=True)
        for image_id, image_url in variation["display_images"].items():
            dst_path = os.path.join(dst_dir, f"{image_id}_{os.path.split(image_url)[-1]}")
            if os.path.exists(dst_path):
                continue
            downloaded = False
            for attempt in range(5):
                try:
                    r = requests.get(image_url, headers=STYLE_SPACE_APP_HEADERS)
                except Exception as e:
                    print(f"Got exception {e} for url {image_url}. Attempt {attempt}. Will retry...")
                    sleep(2 ** attempt)
                else:
                    if r.status_code == 200:
                        with open(dst_path, "wb") as f:
                            f.write(r.content)
                        downloaded = True
                        break
                    else:
                        print(f"Got status code {r.status_code} for url {image_url}. Attempt {attempt}. Will retry...")
                        sleep(2 ** attempt)
            if not downloaded:
                print(f"Failed to download {image_url}; variation_id={variation_id}; id={variation['id']}")


def main():
    if os.path.exists(PRODUCTS_FILE):
        with open(PRODUCTS_FILE) as f:
            products_collection = json.load(f)
    else:
        search_products_args = {
            "page_size": "5000",
            "search_text": "",
            "sort_by": "time",
            "page": None,
        }

        products_collection = {}

        start_ts = time()
        page = 0
        while True:
            search_products_args.update(
                {
                    "page": page,
                }
            )
            search_products_args_query_string = "&".join([f"{k}={v}" for k, v in search_products_args.items()])
            print(
                page,
                len(products_collection),
                (time() - start_ts) / (len(products_collection) if products_collection else 1),
            )

            r = requests.get(
                "https://app.tryoncloth.com/search_product?" + search_products_args_query_string,
                headers=STYLE_SPACE_APP_HEADERS,
            )
            if r.status_code == 200:
                products_list = r.json()["products"]
                if products_list:
                    # variations_count_before = len(products_collection)
                    products_collection.update({p["variation_id"]: p for p in products_list})
                    page += 1
                else:
                    break
            else:
                print(r.status_code, r.content)
                try:
                    pprint(r.json())
                except Exception:
                    pass

        with open(PRODUCTS_FILE, "w") as f:
            json.dump(products_collection, f)

    # products_collection_tops_bottoms = {
    #     key: value for key, value in products_collection.items()
    #     if value["tryon"]["category"] in ("tops", "bottoms")
    # }
    # products_collection_other = {
    #     key: value for key, value in products_collection.items()
    #     if key not in products_collection_tops_bottoms
    # }

    # cpu_count = multiprocessing.cpu_count()
    #
    # for collection in (products_collection_tops_bottoms, products_collection_other):
    #     products_count = len(collection)
    #     print("Downloading", set((item["tryon"]["category"] for item in collection.values())), f": {products_count}")
    #     with multiprocessing.Pool(cpu_count * 32) as pool:
    #         for idx, _none in enumerate(pool.imap_unordered(fetch_images, collection.items())):
    #             if not idx % 1000:
    #                 print(f"{idx} / {products_count}")

    products_categories = {value["tryon"]["category"] for value in products_collection.values()}
    items_to_show = 1000
    for category in products_categories:
        collection = {
            key: value for key, value in products_collection.items() if value["tryon"]["category"] == category
        }
        cnt = 0
        with open(os.path.join(PRODUCTS_DIR, str(category), "index.html"), "w") as f:
            f.write("<table>")
            for key, item in collection.items():
                for variation in item["media_metadata"]:
                    f.write("<tr>")
                    dst_dir = os.path.join(PRODUCTS_DIR, str(category), key, variation["id"])
                    os.makedirs(dst_dir, exist_ok=True)
                    for image_id in variation["display_images_order"]:
                        image_url = variation["display_images"][image_id]
                        image_path = os.path.join(dst_dir, f"{image_id}_{os.path.split(image_url)[-1]}")
                        f.write(
                            f"<td>{image_id}"
                            f"<img src='../../{image_path}' style='max-width: 200px; max-height: 200px'/>"
                            f"</td>"
                        )
                    f.write("</tr>")
                cnt += 1
                if cnt > items_to_show:
                    break
            f.write("</table>")


if __name__ == "__main__":
    main()
