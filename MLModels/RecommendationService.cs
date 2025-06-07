using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace practica4.MLModels
{
    public class RecommendationService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<ProductRating, ProductRatingPrediction> _predictionEngine;
        private List<string> _productIds;

        public RecommendationService()
        {
            _mlContext = new MLContext();
            EntrenarModelo();
        }

        private void EntrenarModelo()
        {
            var dataPath = Path.Combine("Data", "ratings-data.csv");

            var dataView = _mlContext.Data.LoadFromTextFile<ProductRating>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            // Obtener productos únicos para recomendaciones posteriores
            _productIds = _mlContext.Data.CreateEnumerable<ProductRating>(dataView, reuseRowObject: false)
                                        .Select(x => x.ProductId)
                                        .Distinct()
                                        .ToList();

            // Construir pipeline con MapValueToKey para UserId y ProductId
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "UserIdEncoded", inputColumnName: nameof(ProductRating.UserId))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "ProductIdEncoded", inputColumnName: nameof(ProductRating.ProductId)))
                .Append(_mlContext.Recommendation().Trainers.MatrixFactorization(new MatrixFactorizationTrainer.Options
                {
                    MatrixColumnIndexColumnName = "UserIdEncoded",
                    MatrixRowIndexColumnName = "ProductIdEncoded",
                    LabelColumnName = nameof(ProductRating.Label),
                    NumberOfIterations = 20,
                    ApproximationRank = 100
                }));

            _model = pipeline.Fit(dataView);

            _predictionEngine = _mlContext.Model.CreatePredictionEngine<ProductRating, ProductRatingPrediction>(_model);
        }

        public List<(string productId, float score)> Recomendar(string userId, int max = 5)
        {
            var resultados = new List<(string productId, float score)>();

            foreach (var productId in _productIds)
            {
                var prediction = _predictionEngine.Predict(new ProductRating
                {
                    UserId = userId,
                    ProductId = productId
                });

                resultados.Add((productId, prediction.Score));
            }

            return resultados.OrderByDescending(r => r.score).Take(max).ToList();
        }
    }
}
