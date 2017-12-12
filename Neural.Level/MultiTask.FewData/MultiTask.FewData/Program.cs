using ShellProgressBar;
using System;
using System.Threading;
using VI.Neural.Factory;
using VI.Neural.Node;
using VI.NumSharp;

namespace MultiTask.FewData
{
    class Program
    {
                                      //              Parameters that describe one function
        private static float a = .7f; //              F1 = F(W)          = ( (A * W)                               ) (taskA)
        private static float b = .1f; //              F2 = F(W, X)       = ( (A * W) + (B * X)                     ) (taskB)              
        private static float c = .5f; //              F3 = F(W, X, Y)    = ( (A * W) + (B * X) + (C * Y)           ) (taskC)
        private static float d = .3f; //              F4 = F(W, X, Y, Z) = ( (A * W) + (B * X) + (C * Y) + (D * Z) ) (taskD)
                                      //              Our target function with Few Data are: F1 and F3

        private static int epoch = 5;

        private static Random rd = new Random();

        private static volatile INeuron shared;
        private static volatile INeuron taskA;
        private static volatile INeuron taskB;
        private static volatile INeuron taskC;
        private static volatile INeuron taskD;

        static void Main(string[] args)
        {
            ProcessingDevice.Device = Device.CUDA;
                                  
            Console.WriteLine("First Test -- Training Function F1 with 10 samples");
            InitNodes();

            var dataTrain1 = GenerateData(1, 3);
            var dataTest1 = GenerateData(1, 20);

            RunTraining(dataTrain1.input, dataTrain1.desired, dataTrain1.size, RunTrainingFirst);
            var errorFirstTestF1 = RunTest(dataTrain1.input, dataTrain1.desired, dataTrain1.size, RunTestFirst);

            Console.WriteLine($"First Test Error = {errorFirstTestF1}");

            Console.WriteLine("\n---------------------------------------------------------------------------------------------\n");
            Console.WriteLine("First Test -- Training Function F3 with 10 samples");
            InitNodes();

            var dataTrain2 = GenerateData(3, 3);
            var dataTest2 = GenerateData(3, 20);

            RunTraining(dataTrain1.input, dataTrain2.desired, dataTrain2.size, RunTrainingFirst);
            var errorFirstTestF3 = RunTest(dataTrain2.input, dataTrain2.desired, dataTrain2.size, RunTestFirst);
            
            Console.WriteLine($"First Test Error = {errorFirstTestF3}");

            Console.WriteLine("\n---------------------------------------------------------------------------------------------\n");
            Console.WriteLine("Second Test -- 100 F4, 10 F3 and 10 f1");
            InitNodes();

            var errorSecondTestF1 = 0d;
            var errorSecondTestF3 = 0d;

            Console.WriteLine($"Second Test Error:\nF1 = {0}\nF3 = {0}");

            Console.WriteLine("\n---------------------------------------------------------------------------------------------\n");
            Console.WriteLine("Third Test -- 100 F4, 100 F2, 10 F1 and 10 F3");
            InitNodes();

            var errorThirdTestF1 = 0d;
            var errorThirdTestF3 = 0d;

            Console.WriteLine($"Third Test Error:\nF1 = {0}\nF2 = {0}\nF3 = {0}");
                       
            Console.WriteLine($"\n\n\n");

            Console.WriteLine("Function F1 Results:");
            Console.WriteLine($"Fist Test: {errorFirstTestF1} <=> Second Test: {errorSecondTestF1} <=> Third Test: {errorThirdTestF1}\n");

            Console.WriteLine("Function F3 Results:");
            Console.WriteLine($"Fist Test: {errorFirstTestF3} <=> Second Test: {errorSecondTestF3} <=> Third Test: {errorThirdTestF3}");

            Console.WriteLine($"\n\n\n");

            Console.ReadKey();
        }

        #region DataGenerate
        private static (float[][] input, float[] desired, int size) GenerateData(int size, int count)
        {
            float[][] input = new float[count][];
            float[] desired = new float[count];

            for (int i = 0; i < count; i++)
            {
                input[i] = new float[4];
                for (int j = 0; j < size; j++)
                {
                    input[i][j] = (float)rd.NextDouble();
                }
            }


            for (int i = 0; i < count; i++)
            {
                desired[i] = input[i][0] * a +
                             input[i][1] * b +
                             input[i][2] * c +
                             input[i][3] * d;
            }

            return (input, desired, count);
        }
        #endregion

        #region Auxiliar Methods

        private static void InitNodes()
        {
            shared = null;
            shared = new LayerCreator(3, 4)
              .WithLearningRate(.01f)
              .WithMomentum(0)
              .FullSynapse(-3f)
              .Supervised()
              .DenseLayer()
              .WithSigmoid()
              .WithSgd()
              .Hidden()
              .Build();

            taskA = null;
            taskA = new LayerCreator(1, 3)
              .WithLearningRate(.1f)
              .WithMomentum(0)
              .FullSynapse(5f)
              .Supervised()
              .DenseLayer()
              .WithSigmoid()
              .WithSgd()
              .Output()
              .Build();

            taskB = null;
            taskB = new LayerCreator(1, 3)
               .WithLearningRate(.1f)
               .WithMomentum(0)
               .FullSynapse(5f)
               .Supervised()
               .DenseLayer()
               .WithSigmoid()
               .WithSgd()
               .Output()
               .Build();

            taskC = null;
            taskC = new LayerCreator(1, 3)
              .WithLearningRate(.1f)
              .WithMomentum(0)
              .FullSynapse(5f)
              .Supervised()
              .DenseLayer()
              .WithSigmoid()
              .WithSgd()
              .Output()
              .Build();

            taskD = null;
            taskD = new LayerCreator(1, 3)
              .WithLearningRate(.1f)
              .WithMomentum(0)
              .FullSynapse(5f)
              .Supervised()
              .DenseLayer()
              .WithSigmoid()
              .WithSgd()
              .Output()
              .Build();
        }

        private static void RunTraining(float[][] input, float[] desired, int size, Action<int, float[][], float[]> act)
        {
            var options = new ProgressBarOptions
            {
                ForegroundColor = ConsoleColor.Yellow,
                BackgroundColor = ConsoleColor.DarkYellow,
                ProgressCharacter = '─'
            };
            var childOptions = new ProgressBarOptions
            {
                ForegroundColor = ConsoleColor.Green,
                BackgroundColor = ConsoleColor.DarkGreen,
                ProgressCharacter = '─',
                CollapseWhenFinished = true
            };
            using (var pbar = new ProgressBar(epoch, "Training Epoch", options))
            {
                for (int j = 0; j < epoch; j++)
                {
                    using (var child = pbar.Spawn(size, "Train samples", childOptions))
                    {
                        for (int i = 0; i < size; i++)
                        {
                            act(i, input, desired);
                            child.Tick($"Training sample {child.CurrentTick + 1} of {size}");
                        }
                    }
                    pbar.Tick($"Epoch {pbar.CurrentTick + 1} of {epoch}");
                }
            }
        }
        private static double RunTest(float[][] input, float[] desired, int size, Func<int, float[][], float[], double> act)
        {
            var e = 0d;
            for (int i = 0; i < size; i++)
            {
                e += act(i, input, desired);
            }
            e /= size;
            return e;
        }
        #endregion

        #region Methods
        private static void RunTrainingFirst(int i, float[][] input, float[] desired)
        {
            var inp = input[i];
            var des = new[] { desired[i] };

            var sr = shared.Output(inp);
            var tAr = taskA.Output(sr);

            var _tAerror = ((ISupervisedLearning)taskA).Learn(sr, des);
            var serror = ((ISupervisedLearning)shared).Learn(inp, _tAerror);
        }
        private static double RunTestFirst(int i, float[][] input, float[] desired)
        {
            var inp = input[i];
            var des = new[] { desired[i] };

            var sr = shared.Output(inp);
            var tAr = taskA.Output(sr);

            var e0 = Math.Abs(tAr[0] - des[0]);
            var error = Math.Sqrt(Math.Abs(e0 * e0));
            return error / 2.0;
        }

        private static void RunTrainingSecond(int i, float[][] input, float[] desired)
        {
            var inp = input[i];
            var des = new[] { desired[i] };

            var sr = shared.Output(inp);
            var tCr = taskC.Output(sr);

            var _tCerror = ((ISupervisedLearning)taskC).Learn(sr, des);
            var serror = ((ISupervisedLearning)shared).Learn(inp, _tCerror);
        }
        private static double RunTestSecond(int i, float[][] input, float[] desired)
        {
            var inp = input[i];
            var des = new[] { desired[i] };

            var sr = shared.Output(inp);
            var tCr = taskC.Output(sr);

            var e0 = Math.Abs(tCr[0] - des[0]);
            var error = Math.Sqrt(Math.Abs(e0 * e0));
            return error / 2.0;
        }
        #endregion
    }
}
