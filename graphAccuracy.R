isDogData <- read.csv('isDog.csv')
notDogData <- read.csv('notDog.csv')
isDog <- isDogData[, c("epochs", "prediction")]
notDog <- notDogData[, c("epochs", "prediction")]

# Save image as png
png(filename="linear.png", width=1000, height=800, res=150)

plot(isDog$epochs, isDog$prediction, pch=1, type="p", col="green",xlab="# of epochs", ylab="prediction (0 - 1)", main="Purely Linear Network Predictions")
lines(notDog$epochs, notDog$prediction, pch=2, type="p", col="red")
legend("bottomright", legend=c("isDog", "notDog"), col=c("green", "red"), pch=c(1,2))


dev.off()


