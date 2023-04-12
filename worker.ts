import * as tf from '@tensorflow/tfjs'

const CLASSES = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'wild geranium', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle',
    'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily',
    'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william',
    'carnation', 'garden phlox', 'love in the mist', 'cosmos', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose',
    'barberton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'daisy', 'common dandelion',
    'petunia', 'wild pansy', 'primula', 'sunflower', 'lilac hibiscus', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia',
    'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'iris', 'windflower', 'tree poppy',
    'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium',
    'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily',
    'hippeastrum ', 'bee balm', 'pink quill', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower',
    'trumpet creeper', 'blackberry lily', 'common tulip', 'wild rose']

let model: tf.GraphModel | null = null;

async function warmUpModel(model) {
    const warmUpInputs = tf.zeros([1, 192, 192, 3]);
    const warmUpOutput = model.predict(warmUpInputs);
    await warmUpOutput.data();
    warmUpInputs.dispose();
    warmUpOutput.dispose();
}

const displayPercent = (percent) => `${(percent * 100).toFixed(2)}%`

const progressCallback = (f) => self.postMessage({ type: "progress", frac: f})

tf.loadGraphModel('./flowerModelJS/model.json', {onProgress: progressCallback}).then(async (m) => {
    model = m;
    await warmUpModel(m);
    self.postMessage({ type: "ready" })
})

self.addEventListener('message', (e) => {
    if (e.data.type === 'predict') {
        if (model === null) {
            return;
        }

        const { imageData } = e.data;

        const tensor = tf.tidy(() => {
            const imgTensor = tf.browser.fromPixels(imageData);
            const retTensor = imgTensor.expandDims(0).toFloat();
            return retTensor;
        })

        const predictions = model.execute(tensor) as tf.Tensor;
        const {values, indices} = tf.topk(predictions, 5);
        Promise.all([values.data(), indices.data()]).then(([valuesArray, indicesArray]) => {
            let retString = ""
            for (let i = 0; i < 5; i++) {
                retString += `${CLASSES[indicesArray[i]]} : ${displayPercent(valuesArray[i])} \n`;
            }
            self.postMessage({ type: "result", data: retString});

            predictions.dispose();
            values.dispose();
            indices.dispose();
        })
    }
})