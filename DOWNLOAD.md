Dataset **WIDER FACE** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](Set 'HIDE_DATASET=False' to generate download link)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='WIDER FACE', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be downloaded here:

- [WIDER Face Training Images](https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view?usp=sharing)
- [WIDER Face Validation Images](https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view?usp=sharing)
- [WIDER Face Testing Images](https://drive.google.com/file/d/1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T/view?usp=sharing)
- [Face annotations](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip)
- [Examples and formats of the submissions](http://shuoyang1213.me/WIDERFACE/support/example/Submission_example.zip)
