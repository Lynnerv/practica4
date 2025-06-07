using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using practica4.MLModels;

namespace practica4.Controllers
{
    public class SentimentController : Controller
    {
        private readonly SentimentService _sentimentService;

        public SentimentController()
        {
            _sentimentService = new SentimentService();
        }

        [HttpGet]
        public IActionResult Analizar()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Analizar(string texto)
        {
            var resultado = _sentimentService.Predecir(texto);
            ViewBag.Texto = texto;
            ViewBag.Resultado = resultado.Prediction ? "Positivo" : "Negativo";
            ViewBag.Score = resultado.Score.ToString("0.00");
            return View();
        }
    }
}
