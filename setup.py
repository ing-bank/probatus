from setuptools import setup
import os

try:
    if os.environ.get('CI_COMMIT_TAG'):
    	version = os.environ['CI_COMMIT_TAG']
    else:
        version = os.environ['CI_JOB_ID']
except:
    version = 'local'
setup(
    name='pyrisk',
    version=version,
    description='Validate your models like a lion',
    author='RPAA ING',
    author_email='ml_risk_and_pricing_aa@ing.com',
    license='ING Open Source',
    packages=['pyrisk'],
    url='https://gitlab.com/ing_rpaa/pyrisk',
    zip_safe=False
)
