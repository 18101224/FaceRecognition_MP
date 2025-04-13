from models.Far_Centers import Far_Weight

weight = Far_Weight(dims=512,n_classes=7)
sims = weight.check_opt()
print(sims)