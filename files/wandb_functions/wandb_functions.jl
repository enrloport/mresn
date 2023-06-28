

# default Wandb logging function
function wandb_logger(project; entity="elortiz", name=__time_now())
    wb   = Wandb.wandb
    wb.login(key="194ef17f82d72d15afa2b68cf5bc42bdb7b325e4")
    lg = WandbLogger( project=project, entity=entity, name=name )
    Logging.global_logger(lg)
    return lg
end

# This function logs an artifact to a wandb-logger session
function wandb_log_artif(lg, name, A; type = "model")
    aux = typeof(A) == DataFrame ? A : Tables.table(A)
    CSV.write(name*".csv",  aux, writeheader=false)
    wa = WandbArtifact(name, type = type)
    Wandb.add_file(wa, name*".csv")
    Wandb.log(lg, wa)
end