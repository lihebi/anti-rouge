#!/usr/bin/env python3

from preprocessing import preprocess_USE_story
from preprocessing import preprocess_InferSent_story
from preprocessing import preprocess_InferSent_negative

if __name__ == '__main__':
    # preprocess_USE_story(1000)
    # preprocess_InferSent_story(500)
    preprocess_InferSent_negative(1000)
