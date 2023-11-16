import base64
import boto3
import tempfile
import uuid
from enum import Enum


BUCKET_NAME = "mestyle-app"

class FileType(Enum):
    images = "images"
    models = "models"

class S3Storage:

    def __init__(self) -> None:
        pass

    @staticmethod
    def upload(filename: str, filetype: FileType, base64content: str):
        """
        Upload based64 encoded content to S3 storage bucket
        """
        s3 = boto3.resource("s3")
        s3path = "/".join([filetype.name, filename])
        obj = s3.Object(BUCKET_NAME, s3path)
        obj.put(Body=base64.b64decode(base64content))
        
        return "s3://{bucket}/{path}".format(bucket=BUCKET_NAME, path=s3path)

    @staticmethod
    def download(s3path: str):
        """
        Download object from S3 bucket to local machine
        """
        s3_parts = s3path.split("://")
        bucket = s3_parts[1].split("/")[0]
        s3object = "/".join(s3_parts[1].split("/")[1:])

        s3 = boto3.client("s3")
        temp = tempfile.NamedTemporaryFile()
        '''Download and store in temp file'''
        with open(temp.name, "wb") as f:
            s3.download_fileobj(bucket, s3object, f)

        '''Convert image content to base64 string'''
        with open(temp.name, mode="rb") as f:
            content = f.read()
            return base64.b64encode(content)


if __name__ == "__main__":
    pass
