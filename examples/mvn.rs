// cargo run --example mvn -- --means 0 0 --covariance 1 0.4 0 1 --samples 10000 | gnuplot -e 'plot "-"; pause 100'
use gp::distributions::MultivariateNormal;
use gp::matrix::Matrix;
use gp::vector::ColVec;
use rand::distr::Distribution;

fn main() -> noargs::Result<()> {
    let mut args = noargs::raw_args();
    args.metadata_mut().app_name = env!("CARGO_PKG_NAME");
    args.metadata_mut().app_description = "Sample from a multivariate normal distribution";

    if noargs::VERSION_FLAG.take(&mut args).is_present() {
        println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    noargs::HELP_FLAG.take_help(&mut args);

    let samples: usize = noargs::opt("samples")
        .default("100")
        .take(&mut args)
        .then(|a| a.value().parse())?;

    let mut means = Vec::new();
    loop {
        let opt = noargs::opt("means").take(&mut args);
        if !opt.is_present() {
            break;
        }
        means.push(opt.value().parse::<f64>()?);
    }

    let mut covariance = Vec::new();
    loop {
        let opt = noargs::opt("covariance").take(&mut args);
        if !opt.is_present() {
            break;
        }
        covariance.push(opt.value().parse::<f64>()?);
    }

    if let Some(help) = args.finish()? {
        print!("{help}");
        return Ok(());
    }

    let means = ColVec::from(means);
    let covariance = Matrix::from_vec(means.len(), means.len(), covariance);
    let mvn = MultivariateNormal::new(means, covariance)?;

    let mut rng = rand::rng();
    for xs in mvn.sample_iter(&mut rng).take(samples) {
        for x in xs.as_slice() {
            print!("{} ", x);
        }
        println!();
    }

    Ok(())
}
