import { context } from 'esbuild';
import * as React from 'react'
import * as ReactDOM from 'react-dom/client'
import Webcam from 'react-webcam'

const displayPercent = (percent) => `${(percent * 100).toFixed(2)}%`

window.onload = () => {

    const worker = new Worker('./worker.js');

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;

    const videoConstraints = {
        width: 1280,
        height: 720,
        facingMode: "environment"
    };

    const App = () => {
        const [retText, setRetText] = React.useState("Nothing yet");
        const [isReady, setReady] = React.useState(false);
        const [progress, setProgress] = React.useState(0);

        React.useEffect(() => {
            // This effect will trigger whenever isReady changes, and will cause the component to re-render.
        }, [isReady]);

        const webcamRef = React.useRef<Webcam>(null);

        worker.onmessage = (e) => {
            if (e.data.type === 'progress') {
                setProgress(e.data.frac)
            }

            if (e.data.type === 'ready') {
                setReady(true);
            }

            if (e.data.type === 'result') {
                const retString = e.data.data;
                setRetText(retString);
            }
        }

        const predict = () => {
            const canvas = webcamRef.current!.getCanvas({width: 192, height: 192})!;
            const ctx = canvas.getContext('2d')!;
            worker.postMessage({ type: "predict", imageData: ctx.getImageData(0, 0, canvas.width, canvas.height)})
        }

        return (
        <>
        <Webcam
            audio={false}
            height={720}
            screenshotFormat="image/jpeg"
            width={1280}
            videoConstraints={videoConstraints}
            ref={webcamRef}
            style={{"display": "block", "width": "100vw", "height": "75vh"}}
        />
        <div style={{"display": "flex", "justifyContent": "center"}}>
            <div>
                <button disabled={!isReady} onClick={predict} style={{"display": "block", "fontSize": "2em", "margin": "0 auto"}}>{(isReady) ? "Predict" : displayPercent(progress)}</button>
                <pre style={{"display": "block", "fontSize": "1em"}}>{retText}</pre>
            </div>
        </div>
        </>
        )
    }

    const root = ReactDOM.createRoot(document.getElementById('root')!);
    root.render(<App/>);
}