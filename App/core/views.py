from django.shortcuts import render,HttpResponse
import cv2 as cv

# Create your views here.
def homepage(request):
    return HttpResponse("Hello World!!")