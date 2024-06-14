from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from rest_framework.decorators import api_view
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import os

# Load the trained model
# model = load_model('siamese_model.h5')
model = load_model('my_model.keras')

# Helper function to preprocess images
def preprocess_image(image):
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@api_view(['POST'])
def verify_faces(request):
    img1 = request.FILES['image1']
    img2 = request.FILES['image2']
    image1 = preprocess_image(load_img(img1, target_size=(224, 224)))
    image2 = preprocess_image(load_img(img2, target_size=(224, 224)))
    prediction = model.predict([image1, image2])
    return JsonResponse({'similarity_score': float(prediction[0][0])})
