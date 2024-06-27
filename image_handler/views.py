from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model

decoder = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

ml_model = load_model('image_handler/ml_model.h5')


# Create your views here.

def home(request):
    return  HttpResponse("yep you are here")

def predict(request):
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES['image']
            image_data = uploaded_file.read()
            image_array = np.frombuffer(image_data, np.uint8)
            img = cv.imdecode(image_array, cv.IMREAD_COLOR)
            if img is None: return JsonResponse({'error': 'Could not decode the image.'}, status=400)

            input_image = cv.resize(img, (224, 224))
            input_image = input_image.astype(np.float32) / 255.0
            input_image = np.expand_dims(input_image, axis=0)
            predictions = ml_model.predict(input_image)

            predicted_class = np.argmax(predictions, axis=1)
            result = predictions.tolist()

            return JsonResponse({'predicted_class': decoder[predicted_class[0]]})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)
