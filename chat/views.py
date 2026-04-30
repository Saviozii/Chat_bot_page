from django.shortcuts import render
import sys
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from Rag_db import perguntar
# Garante que o Django encontra o pipeline_02.py na raiz do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def chat_view(request):
    return render(request, 'chat/chat.html')



@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        pergunta = data.get('pergunta', '')

        if not pergunta:
            return JsonResponse({'erro': 'Pergunta vazia'}, status=400)

        resultado = perguntar(pergunta)
        return JsonResponse(resultado)

    return JsonResponse({'erro': 'Método não permitido'}, status=405)


