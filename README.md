# ee4690
woo super cool project guys!!!!

## Setup guide
```
git@github.com:Ludi-1/ee4690.git # Setup repository
cd ee4690
virtualenv --python=/usr/bin/python3.8 .venv # Setup virtual Python environment
echo 'export PYTHONPATH="$PWD:$PYTHONPATH"' >> .venv/bin/activate
source .venv/bin/activate

pip install -r requirements.txt
```