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
    public class RecommendationController : Controller
    {
        private readonly RecommendationService _service;

        public RecommendationController()
        {
            _service = new RecommendationService();
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Index(string userId)
        {
            var recomendaciones = _service.Recomendar(userId);
            ViewBag.UserId = userId;
            ViewBag.Recomendaciones = recomendaciones;
            return View();
        }
    }
}
