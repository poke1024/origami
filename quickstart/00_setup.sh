#!/bin/bash

# clone origami

git clone https://github.com/poke1024/origami

# download and extract segmentation models from https://www.dropbox.com/home/Forschungsdaten/Permanent/origami/models/segmentation/origami-models-v3

wget -O models.zip https://uc3e52a80f466ae06a2aeaf6177d.dl.dropboxusercontent.com/zip_download_get/BYHm83cJuAkPoSHaOaEuP0Nklj-R2NSiS0s0UuzhOmnmWRZlPVex9F1EonUfHoQQ_WMfqyRnAsiylZcBja91wxwd-ssvBQhSnrfyFkWbjcKaEQ?_download_id=39650021329642565662846846334081210177730464747868812499130017443&_notify_domain=www.dropbox.com

7z x models.zip

# download ocr models from https://www.dropbox.com/home/Forschungsdaten/Permanent/origami/models/ocr/origami-v19-h56-styled-v2

wget -O ocr_models.zip https://uce885e6fe454abc401860d0ab47.dl.dropboxusercontent.com/zip_download_get/BYFm2I3jFmm9D1rKK26rJSFbLKCYRGHlq464PjAfBKyfRB28ygPsexKwei4kjKA9wEa6BCJEx9OPkrQq1S5SMOu5gn5jWqjQ1QCizk-7cYBE6Q?_download_id=51690070418615855778796283804879446316078219205726683540385665394&_notify_domain=www.dropbox.com

7z x ocr_models.zip
