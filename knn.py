import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition as fr
from face_recognition.face_recognition_cli import image_files_in_folder as imgFolder

ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg' }

def train(trainDir, modelSavePath = None, numberNeighbors = None, knnType = 'ball_tree', verbose = False):
  X = []
  y = []

  # Faz um loop por cada pessoa no conjunto de treinamento
  for classDir in os.listdir(trainDir):
    if not os.path.isdir(os.path.join(trainDir, classDir)):
      continue

    # Faz um loop em cada imagem de treinamento para a pessoa atual
    for imgPath in imgFolder(os.path.join(trainDir, classDir)):
      image = fr.load_image_file(imgPath)
      faceBounding = fr.face_locations(image)

      if len(faceBounding) != 1:
        # Se não houver pessoas na imagem de treinamento, pula a imagem.
        if verbose:
          print("Imagem {} não é adequada para treinamento: {}".format(imgPath, "Não encontrou um rosto" if len(faceBounding) < 1 else "Encontrou mais de um rosto"))
      else:
        # Adicione a codificação de rosto da imagem atual
        # ao conjunto de treinamento
        X.append(fr.face_encodings(image, known_face_locations=faceBounding)[0])
        y.append(classDir)

  # Determina quantos vizinhos usar para pesar no classificador KNN
  if numberNeighbors is None:
    numberNeighbors = int(round(math.sqrt(len(X))))
    if verbose:
      print("Escolha automatica para numberNeighbors:", numberNeighbors)

  # Cria e treina o classificador KNN
  knnModel = neighbors.KNeighborsClassifier(n_neighbors = numberNeighbors, algorithm = knnType, weights = 'distance')
  knnModel.fit(X, y)

  # Salva o classificador KNN treinado
  if modelSavePath is not None:
    with open(modelSavePath, 'wb') as f:
      pickle.dump(knnModel, f)

  return knnModel

def predict(filePath, knnModel = None, modelPath = None, distanceThreshold = 0.6):
  # Validando se as extensões das imagens estão corretas
  if not os.path.isfile(filePath) or os.path.splitext(filePath)[1][1:] not in ALLOWED_EXTENSIONS:
    raise Exception("Caminho de imagem invalido: {}".format(filePath))

  # Verifica se está passando um modelo de classificação treinado
  if knnModel is None and modelPath is None:
    raise Exception("Deve fornecer um modelo de classificacao KNN")

  # Carrega um modelo KNN treinado
  if knnModel is None:
    with open(modelPath, 'rb') as f:
      knnModel = pickle.load(f)

  # Carrega a imagem e encontra os locais dos rostos
  image = fr.load_image_file(filePath)
  faceLocations = fr.face_locations(image)

  # Se nenhum rosto for encontrado na imagem, retorne um resultado (array) vazio
  if len(faceLocations) == 0:
    return []

  # Encontre encodings dos rostos na imagem
  facesEncodings = fr.face_encodings(image, known_face_locations = faceLocations)

  # Usa o modelo knn para encontrar as melhores correspondências dos rostos na imagem de teste
  closestDistances = knnModel.kneighbors(facesEncodings, n_neighbors=1)
  areMatches = [closestDistances[0][i][0] <= distanceThreshold for i in range(len(faceLocations))]

  # Prever classes e remove classificações que não estejam dentro do limite
  return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knnModel.predict(facesEncodings), faceLocations, areMatches)]

def showImage(imgPath, predictions):
  pilImage = Image.open(imgPath).convert("RGB")
  draw = ImageDraw.Draw(pilImage)

  for name, (top, right, bottom, left) in predictions:
      # Desenha uma caixa ao redor do rosto
      draw.rectangle(((left, top), (right, bottom)), outline = (0, 0, 255))
      name = name.encode("UTF-8")

      # Adiciona o nome da pasta de treinamento do rosto
      textWidth, textHeight = draw.textsize(name)
      draw.rectangle(((left, bottom - textHeight - 10), (right, bottom)), fill = (0, 0, 255), outline = (0, 0, 255))
      draw.text((left + 6, bottom - textHeight - 5), name, fill = (255, 255, 255, 255))

  del draw

  # Mostra a imagem com o contorno e o nome no rosto
  pilImage.show()

if __name__ == "__main__":
  # ETAPA 1: Treina o classificador KNN e salva o modelo localmente
  print("Treinando o classificador KNN...")
  classifier = train("images/train", modelSavePath = "model.clf", numberNeighbors = 2)
  print("Treinamento completo!")

  # ETAPA 2
  # Usando o classificador treinado,
  # faz previsões para imagens desconhecidas
  for imageFile in os.listdir("images/test"):
    # Carregando todas imagens de teste
    fullFilePath = os.path.join("images/test", imageFile)
    print("Procurando rostos em {}".format(imageFile))

    # Encontra todas as pessoas na imagem
    # usando um modelo de classificador treinado
    predictions = predict(fullFilePath, modelPath="model.clf")

    # Mostra os resultados no console
    for name, (top, right, bottom, left) in predictions:
      print("- Encontrou {} em ({}, {})".format(name, left, top))

    # Exibe o resultado em imagens
    showImage(os.path.join("images/test", imageFile), predictions)
