python_model_check()
{
  if python3 -c "import $1" >/dev/null 2>&1
  then
      echo "1"
  else
      echo "0"
  fi
}

result=`python_model_check twine`

if [ $result == 1 ]
then
  echo "twine already install"
else
  echo "Install twine package"
  pip3 install twine
fi

python3 setup.py sdist build
python3 -m  twine upload dist/*


