// cargo run --example mvn -- --means 0 0 --covariance 1 0.4 0 1 --samples 10000 | gnuplot -e 'plot "-"; pause 100'
use gp::distributions::MultivariateNormal;
use gp::matrix::Matrix;
use gp::vector::ColVec;
use rand::distr::Distribution;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(long)]
    means: Vec<f64>,

    #[structopt(long)]
    covariance: Vec<f64>,

    #[structopt(long, default_value = "100")]
    samples: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    let means = ColVec::from(opt.means);
    let covariance = Matrix::from_vec(means.len(), means.len(), opt.covariance);
    let mvn = MultivariateNormal::new(means, covariance)?;

    let mut rng = rand::rng();
    for xs in mvn.sample_iter(&mut rng).take(opt.samples) {
        for x in xs.as_slice() {
            print!("{} ", x);
        }
        println!();
    }

    Ok(())
}
