// cargo run --example prior -- --xs $(seq 1 100) --length-scale 1 | gnuplot -e 'plot "-" w l; pause 100'
use gp::distributions::GaussianProcessPrior;
use gp::kernels::GaussianKernel;
use gp::means::ZERO_MEAN;
use rand::distr::Distribution;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Opt {
    #[structopt(long)]
    xs: Vec<f64>,

    #[structopt(long, default_value = "0.1")]
    length_scale: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    let kernel = GaussianKernel::new(opt.length_scale);
    let prior = GaussianProcessPrior::new(&opt.xs, ZERO_MEAN, kernel)?;

    let mut rng = rand::rng();
    let ys = prior.sample(&mut rng);
    for (x, y) in opt.xs.iter().zip(ys.as_slice().iter()) {
        println!("{} {}", x, y);
    }

    Ok(())
}
