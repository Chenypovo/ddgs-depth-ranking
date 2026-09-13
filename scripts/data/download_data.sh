#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /etc/network_turbo >/dev/null 2>&1 || true
mkdir -p downloads dataset
/root/miniconda3/bin/python -m venv --system-site-packages download-venv
download-venv/bin/python -m pip install gdown
if ! download-venv/bin/gdown 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g -O downloads/nerf_llff_data.zip; then
  # MatchNeRF authors explicitly publish this as a backup of the original LLFF archive.
  curl -fL --retry 2 --connect-timeout 30 --max-time 3600 --max-filesize 10737418240 'https://huggingface.co/donydchen/matchnerf/resolve/main/nerf_llff_data.zip' -o downloads/nerf_llff_data.zip
  printf '%s\n' 'https://huggingface.co/donydchen/matchnerf/resolve/main/nerf_llff_data.zip' > downloads/source.txt
fi
download-venv/bin/python - <<'PY'
import zipfile, pathlib, json, datetime
archive=pathlib.Path('downloads/nerf_llff_data.zip')
with zipfile.ZipFile(archive) as z:
    files=[i for i in z.infolist() if '/trex/' in i.filename and not i.is_dir()]
    assert files, 'No trex scene in official archive'
    total=sum(i.file_size for i in files)
    assert total < 12*1024**3, 'Unexpected extraction size'
    root=pathlib.Path('dataset').resolve()
    for info in files:
        out=(root/info.filename).resolve()
        assert out.is_relative_to(root)
        z.extract(info,root)
json.dump({'state':'trex_downloaded_not_preprocessed','archive_bytes':archive.stat().st_size,'extracted_bytes':total,'files':len(files),'at':datetime.datetime.now().isoformat(),'source':'https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view'},open('DATA_STATUS.json','w'),indent=2)
PY
