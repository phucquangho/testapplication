package com.example.testapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.util.Log;
import android.util.Pair;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.linkfirebase.FirebaseModelSource;
import com.google.mlkit.common.model.LocalModel;
import com.google.mlkit.common.model.DownloadConditions;
import com.google.mlkit.common.model.RemoteModelManager;
import com.google.mlkit.common.model.CustomRemoteModel;

import com.google.mlkit.vision.label.ImageLabel;
import com.google.mlkit.vision.label.ImageLabeler;
import com.google.mlkit.vision.label.ImageLabeling;
import com.google.mlkit.vision.label.custom.CustomImageLabelerOptions;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognizer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import io.reactivex.Single;
import io.reactivex.disposables.CompositeDisposable;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import timber.log.Timber;

/**
 * Created by nampham on 2019-08-03.
 */

public class MLService {
    private static final String TAG = MLService.class.getSimpleName();
    Context context;
    CompositeDisposable compositeDisposables;

    public MLService(Context context){
        this.context = context;
        initCustomModel();
        compositeDisposables = new CompositeDisposable();
    }

    /**
     * nhan dien bien so
     * @param filePath
     * @param ivWidth
     * @param ivHeight
     * @param isDetectBienSo
     * @param isDetectLoaiXe
     */
    public void nhanDienLoaiXeVaBienSo(String filePath, int ivWidth, int ivHeight,
                                       boolean isDetectBienSo, boolean isDetectLoaiXe){
        mImageMaxWidth = ivWidth;
        mImageMaxHeight = ivHeight;
        Bitmap bitMap = getBitmapFromFile(filePath);


        Timber.tag(TAG).e("prepare runModelInference");
        if (isDetectLoaiXe){
            Timber.tag(TAG).e("nhận diện loai xe");
            runModelInference(bitMap);
        }
        if (isDetectBienSo){
            Timber.tag(TAG).e("nhận diện bien so");
            runTextRecognition(bitMap);
        }
    }

    private static final String PATTERN_LABEL_CAR = " sports car," +
            " minivan," +
            " minibus," +
            " limousine," +
            " trailer truck," +
            " tow truck," + " ambulance," + "car wheel";
    private static final String PATTERN_LABEL_BIKE = "motor scooter,";
    private void handleLabels(List<String> topLabels){
        if (topLabels == null || topLabels.isEmpty()){
            return;
        }
        //kiểm tra xe hơi
        double point = 0;
        String result = "";
        for (String raw : topLabels){
            String[] data = raw.split(":");
            String label = data[0];
            double value = Double.parseDouble(data[1]);
            boolean isCar = PATTERN_LABEL_CAR.contains(label);
            if (isCar && value > point){
                result = "xe_hoi";
                point = value;
            }
            boolean isScooter = PATTERN_LABEL_BIKE.contains(label);
            if (isScooter && value > point){
                result = "xe_may";
                point = value;
            }
        }
    }




    /**
     * Name of the model file hosted with Firebase.
     */
    private static final String HOSTED_MODEL_NAME = "cloud_model_1";
    //private static final String LOCAL_MODEL_ASSET = "mobilenet_v1_1.0_224_quant.tflite";
    private static final String LOCAL_MODEL_ASSET = "mobilenet_v2_1.0_224_quant.tflite";
    /**
     * Name of the label file stored in Assets.
     */
    private static final String LABEL_PATH = "labels.txt";
    /**
     * Dimensions of inputs.
     */
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    private static final int DIM_IMG_SIZE_X = 224;
    private static final int DIM_IMG_SIZE_Y = 224;
    /**
     * Number of results to show in the UI.
     */
    private static final int RESULTS_TO_SHOW = 3;
    /**
     * Labels corresponding to the output of the vision model.
     */
    private List<String> mLabelList;

    private final PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float>
                                o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });
    /* Preallocated buffers for storing image data. */
    private final int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

    /**
     * An instance of the driver class to run model inference with Firebase.
     */

    private ImageLabeler labeler;


    private void initCustomModel() {
        mLabelList = loadLabelList(context);

        int[] inputDims = {DIM_BATCH_SIZE, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, DIM_PIXEL_SIZE};
        int[] outputDims = {DIM_BATCH_SIZE, mLabelList.size()};
        DownloadConditions conditions = new DownloadConditions
                .Builder()
                .requireWifi()
                .build();
        LocalModel localSource =
                new LocalModel.Builder()
                        .setAssetFilePath(LOCAL_MODEL_ASSET).build();

        CustomRemoteModel cloudSource = new CustomRemoteModel.Builder
                (new FirebaseModelSource.Builder(HOSTED_MODEL_NAME).build())// You could also specify
                // different conditions
                // for updates
                .build();

        RemoteModelManager manager = RemoteModelManager.getInstance();
        manager.download(cloudSource,conditions).
                addOnSuccessListener(new OnSuccessListener<Void>() {
                    @Override
                    public void onSuccess(Void unused) {

                    }
                });

        manager.getInstance().isModelDownloaded(cloudSource)
                .addOnSuccessListener(new OnSuccessListener<Boolean>() {

                    @Override
                    public void onSuccess(Boolean isDownloaded) {
                        CustomImageLabelerOptions.Builder optionsBuilder;
                        if (isDownloaded) {
                            optionsBuilder = new CustomImageLabelerOptions.Builder(cloudSource);
                        } else {
                            optionsBuilder = new CustomImageLabelerOptions.Builder(localSource);
                        }
                        CustomImageLabelerOptions options = optionsBuilder
                                .setConfidenceThreshold(0.5f)
                                .setMaxResultCount(5)
                                .build();
                        labeler = ImageLabeling.getClient(options);
                    }
                });
    }

    //text from image
    private void runTextRecognition(Bitmap bitmap) {
        Timber.tag(TAG).e("runTextRecognition");
        InputImage image = InputImage.fromBitmap(bitmap, 0);
        TextRecognizer recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
        recognizer.process(image)
                .addOnSuccessListener(
                        new OnSuccessListener<Text>() {
                            @Override
                            public void onSuccess(Text texts) {
                                Timber.tag(TAG).e("runTextRecognition - onSuccess");
                                processTextRecognitionResult(texts);
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                // Task failed with an exception
                                e.printStackTrace();
                                Timber.tag(TAG).e("runTextRecognition - onError: " + e.getMessage());
                            }
                        });
    }

    private void processTextRecognitionResult(Text texts) {
        List<Text.TextBlock> blocks = texts.getTextBlocks();
        if (blocks.size() == 0) {
            Timber.tag(TAG).e("processTextRecognitionResult - size = 0");
            return;
        }
        for (int i = 0; i < blocks.size(); i++) {
            List<Text.Line> lines = blocks.get(i).getLines();
            StringBuilder builder = new StringBuilder();
            for (int j = 0; j < lines.size(); j++) {
                List<Text.Element> elements = lines.get(j).getElements();
                if (j != 0){
                    builder.append("|");
                }
                for (int k = 0; k < elements.size(); k++) {
                    Timber.tag(TAG).e("%s",elements.get(k).getText());
                    builder.append(elements.get(k).getText());
                }
            }

            String value = handleBienSoXe(builder.toString());
        }
    }

    private static final Pattern BIEN_SO_PATTERN = Pattern.compile("^(\\d\\d)\\w?.+");
    private static final Pattern PATTERN = Pattern.compile("\\d{4,5}$");
    private String handleBienSoXe(String text){
        Timber.tag(TAG).e("detect: %s", text);
        String value = text.replace(".","");
        value = value.replace("-","");
        boolean isBienSoXe = BIEN_SO_PATTERN.matcher(value).matches();
        if (!isBienSoXe){
            return "";
        }
        //hiệu chỉnh để lấy chính xác được 4 hoặc 5 số cuối của biển số xe
        //nếu chứa "|" thì là xe máy hoặc xe hơi
        //ngược là chụp xe hơi
        if (value.contains("|")){
            value =  value.split("\\|")[1];

        }
        Matcher machter = PATTERN.matcher(value);
        if (machter.find()) {
            return machter.group(0);
        } else {
            return "";
        }
    }

    /**
     * Reads label list from Assets.
     */
    private List<String> loadLabelList(Context context) {
        List<String> labelList = new ArrayList<>();
        try (BufferedReader reader =
                     new BufferedReader(new InputStreamReader(context.getAssets().open
                             (LABEL_PATH)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labelList.add(line);
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to read label list.", e);
        }
        return labelList;
    }

    //detect objection
    private void runModelInference(Bitmap image) {
        // Create input data.
        InputImage inputImage = InputImage.fromBitmap(image,0);
        Timber.tag(TAG).e("runModelInference 1");
        labeler.process(inputImage)
                .addOnSuccessListener(new OnSuccessListener<List<ImageLabel>>() {
                    @Override
                    public void onSuccess(List<ImageLabel> labels) {
                        // Task completed successfully
                        for (ImageLabel label : labels) {
                            String text = label.getText();
                            float confidence = label.getConfidence();
                            int index = label.getIndex();
                        }
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        // Task failed with an exception
                        e.printStackTrace();
                        Timber.tag(TAG).e("runModelInference 2: " + e.getMessage());
                    }
                });

//            FirebaseModelInputs inputs = new FirebaseModelInputs.Builder().add(imgData).build();
//            // Here's where the magic happens!!
//            mInterpreter
//                    .run(inputs, mDataOptions)
//                    .addOnFailureListener(new OnFailureListener() {
//                        @Override
//                        public void onFailure(@NonNull Exception e) {
//                            e.printStackTrace();
//                            Timber.tag(TAG).e("runModelInference 2: " + e.getMessage());
//                        }
//                    })
//                    .continueWith(
//                            new Continuation<FirebaseModelOutputs, List<String>>() {
//                                @Override
//                                public List<String> then(Task<FirebaseModelOutputs> task) {
//                                    byte[][] labelProbArray = task.getResult()
//                                            .<byte[][]>getOutput(0);
//                                    List<String> topLabels = getTopLabels(labelProbArray);
//                                    Timber.tag(TAG).e(topLabels.toString());
//                                    handleLabels(topLabels);
//                                    return topLabels;
//                                }
//                            });
    }


    /**
     * Gets the top labels in the results.
     */
    private synchronized List<String> getTopLabels(byte[][] labelProbArray) {
        for (int i = 0; i < mLabelList.size(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(mLabelList.get(i), (labelProbArray[0][i] &
                            0xff) / 255.0f));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        List<String> result = new ArrayList<>();
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            result.add(label.getKey() + ":" + label.getValue());
        }
        Log.d(TAG, "labels: " + result.toString());
        return result;
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    private synchronized ByteBuffer convertBitmapToByteBuffer(
            Bitmap bitmap, int width, int height) {
        ByteBuffer imgData =
                ByteBuffer.allocateDirect(
                        DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y,
                true);
        imgData.rewind();
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0,
                scaledBitmap.getWidth(), scaledBitmap.getHeight());
        // Convert the image to int points.
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.put((byte) ((val >> 16) & 0xFF));
                imgData.put((byte) ((val >> 8) & 0xFF));
                imgData.put((byte) (val & 0xFF));
            }
        }
        return imgData;
    }

    public  Bitmap getBitmapFromFile(String filePath) {
        InputStream is;
        Bitmap bitmap = null;
        try {
            is = new FileInputStream(new File(filePath));
            bitmap = BitmapFactory.decodeStream(is);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (bitmap != null) {
            // Get the dimensions of the View
            Pair<Integer, Integer> targetedSize = getTargetedWidthHeight();

            int targetWidth = targetedSize.first;
            int maxHeight = targetedSize.second;

            // Determine how much to scale down the image
            float scaleFactor =
                    Math.max(
                            (float) bitmap.getWidth() / (float) targetWidth,
                            (float) bitmap.getHeight() / (float) maxHeight);

            Bitmap resizedBitmap =
                    Bitmap.createScaledBitmap(
                            bitmap,
                            (int) (bitmap.getWidth() / scaleFactor),
                            (int) (bitmap.getHeight() / scaleFactor),
                            true);
            Matrix matrix = new Matrix();
            matrix.postRotate(0);
            Bitmap rotatedBitmap = Bitmap.createBitmap(resizedBitmap, 0, 0,
                    resizedBitmap.getWidth(), resizedBitmap.getHeight(), matrix, true);

            bitmap = rotatedBitmap;
        }
        return bitmap;
    }

    // Max width (portrait mode)
    private Integer mImageMaxWidth;
    // Max height (portrait mode)
    private Integer mImageMaxHeight;

    // Gets the targeted width / height.
    private Pair<Integer, Integer> getTargetedWidthHeight() {
        int targetWidth;
        int targetHeight;
        int maxWidthForPortraitMode = mImageMaxWidth;
        int maxHeightForPortraitMode = mImageMaxHeight;
        targetWidth = maxWidthForPortraitMode;
        targetHeight = maxHeightForPortraitMode;
        return new Pair<>(targetWidth, targetHeight);
    }
}
