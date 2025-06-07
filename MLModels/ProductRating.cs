using Microsoft.ML.Data;

namespace practica4.MLModels
{
    public class ProductRating
    {
        [LoadColumn(0)]
        public string UserId { get; set; }

        [LoadColumn(1)]
        public string ProductId { get; set; }

        [LoadColumn(2)]
        public float Label { get; set; }
    }

    public class ProductRatingPrediction
    {
        public float Score { get; set; }
    }
}
