// cargo run --example prior -- --xs $(seq 1 100) --length-scale 1 | gnuplot -e 'plot "-" w l; pause 100'
use gp::distributions::GaussianProcessPrior;
use gp::kernels::GaussianKernel;
use gp::means::ZERO_MEAN;
use rand::distr::Distribution;

fn main() -> noargs::Result<()> {
    let mut args = noargs::raw_args();
    args.metadata_mut().app_name = env!("CARGO_PKG_NAME");
    args.metadata_mut().app_description = "Sample from a Gaussian process prior";

    if noargs::VERSION_FLAG.take(&mut args).is_present() {
        println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    noargs::HELP_FLAG.take_help(&mut args);

    let length_scale: f64 = noargs::opt("length-scale")
        .default("0.1")
        .take(&mut args)
        .then(|a| a.value().parse())?;

    let mut xs = Vec::new();
    loop {
        let opt = noargs::opt("xs").take(&mut args);
        if !opt.is_present() {
            break;
        }
        xs.push(opt.value().parse::<f64>()?);
    }

    if let Some(help) = args.finish()? {
        print!("{help}");
        return Ok(());
    }

    let kernel = GaussianKernel::new(length_scale);
    let prior = GaussianProcessPrior::new(&xs, ZERO_MEAN, kernel)?;

    let mut rng = rand::rng();
    let ys = prior.sample(&mut rng);
    for (x, y) in xs.iter().zip(ys.as_slice().iter()) {
        println!("{} {}", x, y);
    }

    Ok(())
}
