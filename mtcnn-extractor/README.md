# mtcnn-extractor
## Overview
This is a reference repository to use mtcnn to extract face for QR code.
This service provides a Face extractor implementation of Bio-SDK REST Service. It has to loads [FaceSDKImpl](https://github.com/mosip/mtcnn-extractor) internally on the startup and exposes the endpoints to perform extraction as per the [IBioAPI](https://github.com/mosip/commons/blob/master/kernel/kernel-biometrics-api/src/main/java/io/mosip/kernel/biometrics/spi/IBioApi.java).

To know more about Biometric SDK, refer [biometric-sdk](https://docs.mosip.io/1.2.0/biometrics/biometric-sdk).

## Inspired By
This mtcnn module used for face extraction is inspired by https://github.com/samylee/Mtcnn_Java.

### License
This project is licensed under the terms of [Mozilla Public License 2.0](LICENSE).

