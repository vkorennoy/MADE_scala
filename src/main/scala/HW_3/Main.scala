package HW_3

import breeze.linalg._
import breeze.numerics.pow
import java.io._


object Main {
  val TRAIN_PATH = "C:/Users/And_then_i_woke_up/IdeaProjects/MADE_scala/src/data/train.csv"
  val TEST_PATH = "C:/Users/And_then_i_woke_up/IdeaProjects/MADE_scala/src/data/test.csv"

  final def main(args: Array[String]): Unit = {
    val init_data = read_file()
    val (data, target) = set_target(init_data)
    val (train_data, validate_data, train_target, validate_target) = split(data, target)
    val (_, weights, _) = train(train_data, train_target)
    val (_, validation_score) = validate(validate_data, validate_target, weights)

    val test_init_data = read_file(false)
    val (test_data, test_target) = set_target(test_init_data)
    val (predictions, test_score) = validate(test_data, test_target, weights)
    save_answer_to_file(validation_score, test_score)
    save_predictions_to_file(predictions)
  }

  def train(data: DenseMatrix[Double], target: DenseVector[Double]): (DenseVector[Double], DenseVector[Double], Double) = {
    val weights = inv(data.t * data) * data.t * target
    val target_est = data * weights
    val mse = sum(pow(target_est - target, 2)) / target.length
    (target_est, weights, mse)
  }

  def validate(data: DenseMatrix[Double], target: DenseVector[Double], weights: DenseVector[Double]): (DenseVector[Double], Double) = {
    val target_est = data * weights
    val mse = sum(pow(target_est - target, 2)) / target.length
    (target_est, mse)
  }

  def split(data: DenseMatrix[Double], target: DenseVector[Double], k: Double = 0.2): (
    DenseMatrix[Double], DenseMatrix[Double], DenseVector[Double], DenseVector[Double]
    ) = {
    val k_int: Int = (target.length * k).toInt
    val data_train = data(0 until (k_int - 1), ::)
    val data_validate = data(k_int until target.length, ::)

    val target_train = target(0 until (k_int - 1))
    val target_validate = target(k_int until target.length)
    (data_train, data_validate, target_train, target_validate)
  }

  def read_file(train: Boolean = true): DenseMatrix[Double] = {
    val path:String = if (train) { TRAIN_PATH } else { TEST_PATH }
    val train_matrix = csvread(new File(path), ',', skipLines = 1)
    train_matrix
  }

  def set_target(init_data: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double]) = {
//    println(init_data.rows, init_data.cols)
    val data_size: Int = init_data.cols
    val target = init_data(::, data_size - 1)
    val data =  init_data(::, 0 to(data_size - 2))
    (data, target)
  }

  def save_answer_to_file(val_score: Double, test_score: Double, file_name: String = "score.txt"): Unit = {
    val file = new File(file_name)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(s"Validation MSE score: $val_score. Test MSE score: $test_score" )
    bw.close()
  }

  def save_predictions_to_file(target: DenseVector[Double], file_name: String = "predictions.txt"): Unit = {
    val file = new File(file_name)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(target.toString)
    bw.close()
  }
}
