import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from tqdm.auto import tqdm

# sklearn.datasets.make_moons¶

def create_ad_hoc_dataset(plot=True):
    adhoc_dimension = 2
    train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
        training_size=50,
        test_size=5,
        n=adhoc_dimension,
        gap=0.3,
        plot_data=False,
        one_hot=False,
        include_sample_total=True,
    )

    if plot:

        plt.figure(figsize=(5, 5))
        plt.ylim(0, 2 * np.pi)
        plt.xlim(0, 2 * np.pi)
        plt.imshow(
            np.asmatrix(adhoc_total).T,
            interpolation="nearest",
            origin="lower",
            cmap="RdBu",
            extent=[0, 2 * np.pi, 0, 2 * np.pi],
        )

        plt.scatter(
            train_features[np.where(train_labels[:] == 0), 0],
            train_features[np.where(train_labels[:] == 0), 1],
            marker="s",
            facecolors="w",
            edgecolors="b",
            label="A train",
        )
        plt.scatter(
            train_features[np.where(train_labels[:] == 1), 0],
            train_features[np.where(train_labels[:] == 1), 1],
            marker="o",
            facecolors="w",
            edgecolors="r",
            label="B train",
        )
        plt.scatter(
            test_features[np.where(test_labels[:] == 0), 0],
            test_features[np.where(test_labels[:] == 0), 1],
            marker="s",
            facecolors="b",
            edgecolors="w",
            label="A test",
        )
        plt.scatter(
            test_features[np.where(test_labels[:] == 1), 0],
            test_features[np.where(test_labels[:] == 1), 1],
            marker="o",
            facecolors="r",
            edgecolors="w",
            label="B test",
        )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.title("Ad hoc dataset for classification")

        plt.show()
    
    return (train_features, train_labels), (test_features, test_labels)

class SVM:
    def __init__(self):
        self.lamda = 0.01; # regularization parameter
        self.lr = 1e-5
        self.losses = []
        self.gamma = 0.5

    
    def train(self, X, y, n_iters=1000):
        self.X_train = X
        for _ in tqdm(range(n_iters)):
            # self.fit_step(X, y); 
            self.fit_step_linear(X, y)


    """
    Kernel Explaination:
        We have Eqn Y = Xw + b
        We convert RHS into kernelized form with K_x such that K_x = K (X, X(i)
            so evidently we sum it over all such values
        Y = \sum alpha_i K(X, X(i)) + b
    """
    def kernel(self, X_1, X_2):
        # K(i, j) = RBF(X_i, X_j)
        # RBF = exp(-gamma * ||X_i - X_j||^2)
        K = np.zeros((len(X_1), len(X_2))); 

        # for i in range(len(X)):
        #     for j in range(len(X)):
        #         K[i, j] = RBF_kernel(X[i], X[j])

        # distances = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T); 
        distances = np.sum(X_1**2, axis=1).reshape(-1, 1) + np.sum(X_2**2, axis=1) - 2 * np.dot(X_1, X_2.T); 

        return np.exp(-self.gamma * distances); 

    def fit_step_rbf(self, X, y):
        self.alpha = np.random.randn(len(X), 1)
        
        y_pred = np.dot(self.kernel(X, X), self.alpha) 
        y_pred = np.array(y_pred.reshape(-1), dtype=np.int64); 

        y = np.array(y.reshape(-1), dtype=np.int); 

        sign_vector = np.multiply(y_pred, y);  
        vfunc = np.vectorize(lambda x: 1 if x >= 1 else 0)
        vfunc(sign_vector)
        
        # vectorize:
        # sign_vector = [1, 0, 1, 0, 0] 0-=> sign is -ve
        # y[1, 0, 1, 0, 0] = [y1, y3] : passing bool array into the vector returns the true elements
        a = self.lamda * np.sum(np.square(self.alpha))

        print(np.dot(y[sign_vector].T, y_pred[sign_vector]).shape)
        # b = np.maximum(0, np.dot(y[sign_vector].T, y_pred[sign_vector]))
        b = np.minimum(0, y * y_pred)
        print(b.shape)

        # 1XN Nx1 = 1X1
        # Nx1, Nx1 => Nx1 => max(0, NX1)
        print(b)
        loss = a + b; # loss = \lambda ||\alpha^||2 + \sum max(0, y_i * y_pred_i) 
        # (2nd term checks if pred is correct) 
        self.losses.append(loss)

        dalpha_l = self.lamda * 2 * self.alpha + np.dot(y[sign_vector].T, self.kernel(X, X)).reshape(-1, 1)
        # 2nd term: such that y_pred = \alpha K

        self.alpha -= self.lr * dalpha_l


        # \sum alpha_i * K[:, i]
        # numpy: np.sum(alpha * K[:,], axis=)
        # y_pred = np.dot()
    
    def fit_step_linear(self, X, y):
        # X = (N, D)
        # Y = (N, 1)
        # y_pred = (N, 1)
        # y_pred = Xw + b
        # w = (D, 1)
        """
        Manipulation
        w = (D, 1)
        X = (N, D)
        X_i = (1, D)
          = \alpha_i X X_i T <- K_x
        
        """
        

        self.w = np.random.randn(X.shape[1], 1); 
        self.b = np.random.randn(1); 

        y_pred = np.dot(X, self.w) + self.b; # ypred = Xw + b
        y_pred = np.array(y_pred.reshape(-1), dtype=np.int64); 

        y = np.array(y.reshape(-1), dtype=np.int); 

        sign_vector = np.multiply(y_pred, y);  
        vfunc = np.vectorize(lambda x: 0 if x >= 1 else 1)
        sign_vector = vfunc(sign_vector)
        
        # vectorize:
        # sign_vector = [1, 0, 1, 0, 0] 0-=> sign is -ve
        # y[1, 0, 1, 0, 0] = [y1, y3] : passing bool array into the vector returns the true elements
        a = self.lamda * np.sum(np.square(self.w))
        b = np.sum(np.maximum(0, -y[sign_vector] * y_pred[sign_vector]))
        # print(b)
        loss = a + b; 

        dw_l = self.lamda * 2 * self.w - np.dot(X[sign_vector, :].T, y[sign_vector].reshape(-1, 1))
        db_l = -np.sum(y[sign_vector]); 

        # updates to the weights:
        self.w -= self.lr * dw_l; 
        self.b -= self.lr * db_l; 
    
        # Plot Thickens
        self.losses.append(loss)

    
    def predict_linear(self, X):
        y_pred = np.dot(X, self.w) + self.b; 
        
        return_pred = np.ones(y_pred.shape)
        return_pred[y_pred < 0] = -1

        return return_pred
    
    def predict_rbf(self, X):
        """
        axb = colsxrows
    
        alpha : (Nx1)
        TestK : (N_test, N)
        TestK = [k(x_test_1, x_train_1), k(x_test_1, x_train_2) ... k(x_test_1, x_train_n)]
                [k(x_test_2, )]
        
        
        """
        y_pred = np.dot(self.kernel(X, self.X_train), self.alpha) 
        return_pred = np.ones(y_pred.shape)
        return_pred[y_pred < 0] = -1

        return return_pred
    
#} END CLASS SVM


def test_train_split(X, y, ratio=0.2):
    rounded_train_size = int(len(X) * (1-ratio)); 
    train_X, train_y = X[:rounded_train_size], y[:rounded_train_size]; 
    test_X, test_y = X[rounded_train_size:], y[rounded_train_size:]; 

    return (train_X, train_y), (test_X, test_y); 
    


if __name__ == "__main__":
    X, y = make_moons(noise=0.3, n_samples=1000); 
    print("X,Y Shape: ", X.shape, y.shape)
    
    y[y == 0] = -1
    print(y)

    (train_X, train_y), (test_X, test_y) = test_train_split(X, y, ratio=0.1); 

    svm_classifier = SVM(); 

    svm_classifier.train(train_X, train_y, n_iters=100); 

    # predict bitch!
    y_pred = svm_classifier.predict_linear(test_X); 


    accuracy = 100 * np.sum(y_pred.reshape(-1) == test_y.reshape(-1)) / len(test_y); 

    print(f"Accuracy : {accuracy}%", accuracy)
    plt.plot(svm_classifier.losses)
    plt.show()

    plt.figure(figsize=(8, 6))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = svm_classifier.predict_linear(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



""":GPT
# Plot the decision boundary
plt.figure(figsize=(8, 6))
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
"""