from concurrent import futures
import base64
from turtle import width

import grpc
import cv2

import backend_pb2
import backend_pb2_grpc


class BackendService(backend_pb2_grpc.BackendServicer):
    def _test_func(self, path):
        image = cv2.imread(path)
        h, w, _ = image.shape
        print(f"image shape: w-{w} h-{h}")

        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_str = base64.b64encode(image)
        return image_str, w, h

    def load_image(self, request, context):
        # load the image from disk
        path = request.path
        print(path)

        image_str, w, h = self._test_func(path=path)

        response_message = backend_pb2.image(img_content=image_str, width=w, height=h)
        return response_message

    def grad_cam(array):
        img = preprocess(array)
        model = model_fun()
        preds = model.predict(img)
        argmax = np.argmax(preds[0])
        output = model.output[:, argmax]
        last_conv_layer = model.get_layer("conv10_thisone")
        grads = K.gradients(output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate(img)
        for filters in range(64):
           conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
          # creating the heatmap
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= np.max(heatmap)  # normalize
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img2 = cv2.resize(array, (512, 512))
        hif = 0.8
        transparency = heatmap * hif
        transparency = transparency.astype(np.uint8)
        superimposed_img = cv2.add(transparency, img2)
        superimposed_img = superimposed_img.astype(np.uint8)
        return superimposed_img[:, :, ::-1]

    def preprocess(array):
        array = cv2.resize(array, (512, 512))
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        array = array / 255
        array = np.expand_dims(array, axis=-1)
        array = np.expand_dims(array, axis=0)
        return array
    
    def predict(array):
    #   1. call function to pre-process image: it returns image in batch format
        batch_array_img = preprocess(array)
    #   2. call function to load model and predict: it returns predicted class and probability
        model = model_fun()
    # model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
        prediction = np.argmax(model.predict(batch_array_img))
        proba = np.max(model.predict(batch_array_img)) * 100
        label = ""
        if prediction == 0:
            label = "bacteriana"
        if prediction == 1:
            label = "normal"
        if prediction == 2:
            label = "viral"
    #   3. call function to generate Grad-CAM: it returns an image with a superimposed heatmap
        heatmap = grad_cam(array)
        return (label, proba, heatmap)
        
    def run_model(self):
        self.label, self.proba, self.heatmap = predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.ANTIALIAS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        print("OK")
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    backend_pb2_grpc.add_BackendServicer_to_server(BackendService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
