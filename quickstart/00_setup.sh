#!/bin/bash

# clone origami

git clone https://github.com/poke1024/origami

# download and extract segmentation models from https://www.dropbox.com/home/Forschungsdaten/Permanent/origami/models/segmentation/origami-models-v3

wget -O models.zip "https://uca58c50515cbfaf0363bb0cb6fe.dl.dropboxusercontent.com/zip_download_get/BBVtLmY7hWNyW_qEJe5Xp9kUgjz6oexJK4mMqZNsCWPBrIXJhpKmNHlI6dmSoB6tNd9cIrN4Cf2fFAMzRysSFEvJqQclDwwJPk7DzgWYACABJQ?_download_id=533591005431019280218829531675511048890558876600740525990494341846&_notify_domain=www.dropbox.com"

7z x models.zip

# download ocr models from https://www.dropbox.com/home/Forschungsdaten/Permanent/origami/models/ocr/origami-v19-h56-styled-v2

wget -O ocr_models.zip https://uc2a86067b2188b835119f164df1.dl.dropboxusercontent.com/zip_download_get/BBXU63ZitR4mvhoJL4RgKf8EXg2lTZurhYXigmytEVdkxmOkDqFtm3GkeK7-jJoDRiOCwvO9Hyzs6OuWXW4lTv6fGzPCwjdKyBWDI0FhHbxW7w?_download_id=9274836462678285607588886091311159303633115785869975384670749405&_notify_domain=www.dropbox.com

7z x ocr_models.zip
