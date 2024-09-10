# Reference Implementation Conventions

## Overview

The MAITE protocols for metrics, models, and datasets allow for a great deal of flexibility, and only define 
specific functionality where needed by the integrations that are nearly universal to the workflow of a model evaluation.

During the course of testing capability development, and particularly when new testing capabilities are developed or 
acquired as part of the JATIC program, we will likely identify integrations where the protocols lack the specificity 
required to smoothly incorporate those new capabilities into our existing test cases (or new ones).

This document serves to declare the specific narrowing of scope that the reference implementation (RI) will use for creating
classes and objects that align to MAITE protocol compliant objects.

!!! note
    These conventions do **not** serve to make the objects not comply with the protocols, and any additional specificity required by these protocols must be compliant to the generalized form of the protocols.

As these conventions are added to the reference implementation, it is possible that the opinions applied here become 
absorbed by the protocols in future releases in MAITE, at which time the reference implementation team will be
responsible for updating the use of conventions in the RI to align with the new protocols, and remove the associated 
section from this document if it is no longer needed because of the addition to MAITE.

## Model Conventions

* Model classes and wrappers created within the reference implementation (and any models that will be evaluated using
the reference implementation) will be built with the underlying model API (pytorch, tensorflow, keras, etc.) object 
accessible under an attribute called `model`. For example, a thinly wrapped `yolov5s` model called `wrapped` will have the
underlying `yolov5s` model accessible via a call to `wrapped.model`.
