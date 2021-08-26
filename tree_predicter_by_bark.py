from tensorflow import keras
import sys
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from keras.preprocessing import image

class_names = ['Casuarina equisetifolia',
               'Barringtonia acutangula',
               'Hevea brasiliensis',
               'Anacardium occidentale',
               'Wrightia',
               'Prunnus',
               'Ficus microcarpa',
               'Khaya senegalensis',
               'Delonix regia',
               'Dalbergia oliveri',
               'Prunus salicina',
               'Erythrina fusca',
               'Carica papaya',
               'Artocarpus altilis',
               'Magnolia alba',
               'Khaya senegalensis A.Juss',
               'Tectona grandis',
               'Adenium species',
               'Ficus racemosa',
               'Eucalyptus',
               'Citrus aurantiifolia',
               'Chrysophyllum cainino',
               'Cocos nucifera',
               'Mangifera',
               'Senna siamea',
               'Psidium guajava',
               'Tamarindus indica',
               'Persea',
               'Adenanthera microsperma',
               'Veitchia merrilli',
               'Terminalia catappa',
               'Acacia',
               'Pterocarpus macrocarpus',
               'Lagerstroemia speciosa',
               'Dipterocarpus alatus',
               'Wrightia religiosa',
               'Melaleuca',
               'Artocarpus heterophyllus',
               'Syzygium nervosum',
               'Melia azedarach',
               'Polyalthia longifolia',
               'Cananga odorata',
               'Musa',
               'Spondias mombin L',
               'Nephelium lappaceum',
               'Hopea',
               'Citrus grandis',
               'Annona squamosa',
               'Cedrus',
               'Gmelina arborea Roxb']


main_model = keras.models.load_model('models')


def prediction_of_tree(url):
    response = requests.get(str(url), stream=True)
    img = Image.open(response.raw)
    test_image = np.expand_dims(image.img_to_array(img), axis=0)
    return class_names[np.argmax(np.round(main_model.predict(test_image.reshape(224, 224, 3)), 4)*100)]


if __name__ == '__main__':
    prediction_of_tree(sys.argv[1])
