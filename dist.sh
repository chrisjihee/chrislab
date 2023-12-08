mamba create --name chrislab python=3.9 -y; mamba activate chrislab;
rm -rf build dist src/*.egg-info;
pip install build; python3 -m build;
pip install twine; python3 -m twine upload dist/*;
rm -rf build dist src/*.egg-info;

sleep 3; clear;
mamba create --name chrislab python=3.9 -y; mamba activate chrislab;
sleep 5; clear;
pip install --upgrade chrislab; pip list;
