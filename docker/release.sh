#!/bin/bash
repo="http://110.16.193.170:50083/repository/pypi-hosted/"

function package() {
    rm -r dist
    python setup.py bdist_wheel
    echo "upload to file repository"
    twine upload --verbose -u deploy -p kDNubrlaK6n6RtzN --repository-url $repo dist/*.whl
}

package