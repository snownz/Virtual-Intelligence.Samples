using System;

namespace MultiLayerPerceptron.EvenOrOdd
{
    class Program
    {
        static void Main(string[] args)
        {
            var rd = new Random();
            var values = new[] { .1f, .002f };

            var watch = System.Diagnostics.Stopwatch.StartNew();

            ProcessingDevice.Device = Device.CPU;

            watch.Stop();
            Console.WriteLine($"Device Time: {watch.ElapsedMilliseconds}ms");

            var hiddens = new LayerCreator(2, 4)
                            .WithLearningRate(values[0])
                            .WithMomentum(0)
                            .FullSynapse()
                            .Supervised()
                            .DenseLayer()
                            .WithLeakRelu()
                            .WithSgd()
                            .Hidden()
                            .Build();

            var hiddens2 = new LayerCreator(2, 2)
                .WithLearningRate(values[0])
                .WithMomentum(0)
                .FullSynapse()
                .Supervised()
                .DenseLayer()
                .WithLeakRelu()
                .WithSgd()
                .Hidden()
                .Build();

            var outputs = new LayerCreator(2, 2)
                .WithLearningRate(values[0])
                .WithMomentum(0)
                .FullSynapse()
                .Supervised()
                .DenseLayer()
                .WithSigmoid()
                .WithSgd()
                .Output()
                .Build();

            var loss = new SquareLossFunction();

            watch = System.Diagnostics.Stopwatch.StartNew();
            watch.Stop();
            Console.WriteLine($"Sinapse Time: {watch.ElapsedMilliseconds}ms");

            var trainingValues = new[]
            {
                new[] {0f, 0f, 0f, 0f},
                new[] {1f, 0f, 0f, 0f},
                new[] {0f, 1f, 0f, 0f},
                new[] {0f, 0f, 1f, 0f},
                new[] {0f, 0f, 0f, 1f},
                new[] {1f, 0f, 0f, 0f},
                new[] {1f, 1f, 0f, 0f},
                new[] {0f, 1f, 1f, 0f},
                new[] {0f, 0f, 1f, 1f},
                new[] {0f, 0f, 0f, 1f},
                new[] {0f, 1f, 0f, 0f},
                new[] {1f, 0f, 1f, 0f},
                new[] {0f, 1f, 0f, 1f},
                new[] {1f, 0f, 1f, 0f},
                new[] {0f, 1f, 0f, 1f},
                new[] {0f, 0f, 1f, 0f},
                new[] {1f, 0f, 0f, 1f},
                new[] {1f, 1f, 0f, 0f},
                new[] {0f, 1f, 1f, 0f},
                new[] {0f, 0f, 1f, 1f},
                new[] {1f, 1f, 1f, 0f},
                new[] {1f, 1f, 1f, 1f}
            };

            var desiredValues = new[]
            {
                new[] {1f, 0f},
                new[] {1f, 0f},
                new[] {1f, 0f},
                new[] {1f, 0f},
                new[] {0f, 1f},
                new[] {1f, 0f},
                new[] {1f, 0f},
                new[] {1f, 0f},
                new[] {0f, 1f},
                new[] {0f, 1f},
                new[] {1f, 0f},
                new[] {1f, 0f},
                new[] {0f, 1f},
                new[] {1f, 0f},
                new[] {0f, 1f},
                new[] {1f, 0f},
                new[] {0f, 1f},
                new[] {1f, 0f},
                new[] {1f, 0f},
                new[] {0f, 1f},
                new[] {1f, 0f},
                new[] {0f, 1f}
            };

            int cont = 0;
            int sizeTrain = 10;

            var e = double.MaxValue;

            while (true)
            {
                watch = System.Diagnostics.Stopwatch.StartNew();
                e = 0;
                for (int i = 0; i < sizeTrain; i++)
                {
                    var index = i;// rd.Next(0, trainingValues.Length);
                    var inputs = trainingValues[index];
                    var desireds = desiredValues[index];

                    //watch = System.Diagnostics.Stopwatch.StartNew();
                    // Feed Forward
                    var _h = hiddens.Output(inputs);
                    var _h2 = hiddens2.Output(_h);
                    var _o = outputs.Output(_h2);
                    //watch.Stop();
                    //Console.WriteLine($"\nForward Time: { watch.ElapsedMilliseconds}ms");
                    //Thread.Sleep(100);

                    //watch = System.Diagnostics.Stopwatch.StartNew();
                    // Backward
                    var _oe = ((ISupervisedLearning)outputs).Learn(_h2, desireds);
                    var _he2 = ((ISupervisedLearning)hiddens2).Learn(_h, _oe);
                    ((ISupervisedLearning)hiddens).Learn(inputs, _he2);
                    //watch.Stop();
                    //Console.WriteLine($"\nBackward Time: { watch.ElapsedMilliseconds}ms");

                    // Error
                    var e0 = Math.Abs(_o[0] - desireds[0]);
                    var e1 = Math.Abs(_o[1] - desireds[1]);
                    var error = Math.Sqrt(Math.Abs(e0 * e0 + e1 * e0));
                    e += error / 2.0;
                }

                e /= sizeTrain;
                cont++;
                watch.Stop();
                var time = watch.ElapsedMilliseconds;
                Console.WriteLine($"Interactions: {cont}\nError: {e}");
                //Console.WriteLine($"Interactions: {cont}\nError: {e}\nTime: {time / (double)sizeTrain}ms");
                Console.Title =
                    $"TSPS (Training Sample per Second): {Math.Ceiling(1000d / ((double)time / (double)sizeTrain))}";
            }
    }
}
