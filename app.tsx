import { context } from 'esbuild';
import * as React from 'react'
import * as ReactDOM from 'react-dom/client'
import Webcam from 'react-webcam'
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
    const warmUpInputs = tf.zeros([1, 512, 512, 3]);
    const warmUpOutput = model.predict(warmUpInputs);
    await warmUpOutput.data();
    warmUpInputs.dispose();
    warmUpOutput.dispose();
}


function predict(imageData) {
    if (model === null)
        return Promise.resolve("");

    const tensor = tf.tidy(() => {
        const imgTensor = tf.browser.fromPixels(imageData);
        const retTensor = imgTensor.expandDims(0).toFloat();
        return retTensor;
    })

    const predictions = model.execute(tensor) as tf.Tensor;
    const { values, indices } = tf.topk(predictions, 5);
    return Promise.all([values.data(), indices.data()]).then(([valuesArray, indicesArray]) => {
        let retString = ""
        for (let i = 0; i < 5; i++) {
            retString += `${CLASSES[indicesArray[i]]} : ${displayPercent(valuesArray[i])} \n`;
        }

        predictions.dispose();
        values.dispose();
        indices.dispose();

        return retString;
    })
}


const displayPercent = (percent) => `${(percent * 100).toFixed(2)}%`

window.onload = () => {

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;

    const videoConstraints = {
        width: 1280,
        height: 720,
        facingMode: "environment"
    };

    const App = () => {
        const [retText, setRetText] = React.useState("");
        const [isReady, setReady] = React.useState(false);
        const [progress, setProgress] = React.useState(0);
        const [isPredicting, setPredicting] = React.useState(false);

        React.useEffect(() => {
            // This effect will trigger whenever isReady changes, and will cause the component to re-render.
        }, [isReady, isPredicting]);

        React.useEffect(() => {

            tf.loadGraphModel('./flowerModelJS/model.json', { onProgress: setProgress }).then(async (m) => {
                model = m;
                await warmUpModel(m);
                setReady(true);
            })

        }, []);

        const webcamRef = React.useRef<Webcam>(null);

        const pressPredict = () => {
            const canvas = webcamRef.current!.getCanvas({ width: 512, height: 512 })!;
            const ctx = canvas.getContext('2d')!;
            setPredicting(true);
            predict(ctx.getImageData(0, 0, canvas.width, canvas.height))?.then((retString) => {
                setRetText(retString);
                setPredicting(false);
            });
        }

        return (
            <>
                <Webcam
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                    ref={webcamRef}
                    style={{ "display": "block", "width": "100vw", "height": "75vh" }}
                />
                <div style={{ "display": "flex", "justifyContent": "center" }}>
                    <div>
                        <button disabled={!isReady || isPredicting} onClick={pressPredict} style={{ "display": "block", "fontSize": "2em", "margin": "0 auto", "position": "relative", "top": "-2em" }}>{(isReady) ? "Predict" : displayPercent(progress)}</button>
                        <pre style={{ "display": "block", "fontSize": "1em", "position": "relative", "top": "-3em", "margin": "0" }}>{(isReady) ? retText : "Loading Model..."}</pre>
                    </div>
                </div>
            </>
        )
    }

    const root = ReactDOM.createRoot(document.getElementById('root')!);
    root.render(<App />);
}