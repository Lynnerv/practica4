using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using System.IO;

namespace practica4.MLModels
{
    public class SentimentService
    {
        private readonly MLContext _mlContext;
        private PredictionEngine<SentimentData, SentimentPrediction> _predictionEngine;

        public SentimentService()
        {
            _mlContext = new MLContext();
            EntrenarModelo();
        }

        private void EntrenarModelo()
        {
            var dataPath = Path.Combine("Data", "sentiment-data.tsv");

            var dataView = _mlContext.Data.LoadFromTextFile<SentimentData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: '\t');

            var pipeline = _mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label", featureColumnName: "Features"));

            var model = pipeline.Fit(dataView);

            _predictionEngine = _mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        }

        public SentimentPrediction Predecir(string texto)
        {
            var input = new SentimentData { Text = texto };
            return _predictionEngine.Predict(input);
        }
    }
}
